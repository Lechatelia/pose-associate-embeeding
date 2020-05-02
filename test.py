import cv2
import torch
import tqdm
import os
import numpy as np
import pickle

from data.coco_pose.ref import ref_dir, flipRef
from utils.misc import get_transform, kpt_affine, resize
from utils.group import HeatmapParser
from util import draw_limbs
import matplotlib.pyplot as plt
valid_filepath = ref_dir + '/validation.pkl'

parser = HeatmapParser(detection_val=0.1)

def refine(det, tag, keypoints):
    """
    Given initial keypoint predictions, we identify missing joints
    det: [17, 128, 128]
    tag: [17, 128, 128, 2]
    keypints: [17， 5]
    """
    if len(tag.shape) == 3: # 一般是[17, 128, 128, 2] 2是翻转检测结果
        tag = tag[:,:,:,None]

    tags = []
    for i in range(keypoints.shape[0]): # 某一个关节点
        if keypoints[i, 2] > 0: # 如果有检测结果
            y, x = keypoints[i][:2].astype(np.int32)
            tags.append(tag[i, x, y]) # 添加这个位置的tags

    prev_tag = np.mean(tags, axis = 0) # 求tag均值[2]
    ans = []

    for i in range(keypoints.shape[0]): # 对应这个关节点
        tmp = det[i, :, :] # [128, 128]
        tt = (((tag[i, :, :] - prev_tag[None, None, :])**2).sum(axis = 2)**0.5 ) # [128, 128]对应于每一个点的tag偏差
        tmp2 = tmp - np.round(tt) # heatmap减去tags的偏差 是够可以用e^(-x)是求解
        # tmp 的最大值可能不属于这个目标，但是这个其他目标的最大值减去tags偏差之后就会很小，甚至出现很多负数，这时候找最大值就是满足tags偏差很小的那些点里面找的最符合的关节点
        x, y = np.unravel_index( np.argmax(tmp2), tmp.shape ) # 找到现在的最大值的位置
        xx = x
        yy = y
        val = tmp[x, y] # heatmap value
        x += 0.5
        y += 0.5
        # 调整一下位置， 稍微偏向峰顶位置 adjust函数
        if tmp[xx, min(yy+1, det.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
            y+=0.25
        else:
            y-=0.25

        if tmp[min(xx+1, det.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
            x+=0.25
        else:
            x-=0.25

        x, y = np.array([y,x])
        ans.append((x, y, val))
    ans = np.array(ans)

    if ans is not None:
        for i in range(17):
            if ans[i, 2]>0 and keypoints[i, 2]==0: # 如果之前没找到这个点，但是refine过程中找到了这个点
                keypoints[i, :2] = ans[i, :2] # 就用refine结果去替代
                keypoints[i, 2] = 1 

    return keypoints

def multiperson(img, func, mode):
    """
    1. Resize the image to different scales and pass each scale through the network
    2. Merge the outputs across scales and find people by HeatmapParser
    3. Find the missing joints of the people with a second pass of the heatmaps
    """
    if mode == 'multi':
        scales = [2, 1., 0.5]
    else:
        scales = [1]

    height, width = img.shape[0:2]
    center = (width/2, height/2)
    dets, tags = None, [] # 存储不同尺度的检测结果
    for idx, i in enumerate(scales): # 对于每一个尺度
        scale = max(height, width)/200
        input_res = max(height, width)
        inp_res = int((i * 512 + 63)//64 * 64)
        res = (inp_res, inp_res)

        mat_ = get_transform(center, scale, res)[:2]
        inp = cv2.warpAffine(img, mat_, res)/255 #[512, 512, 3]

        def array2dict(tmp):
            return { # tmp[0]  [bs,4, 68, 128, 128]
                'det': tmp[0][:,:,:17], #  前16个通道作为 [1,4, 17, 128, 128] 注意这里取了全部stage的输出
                'tag': tmp[0][:,-1, 17:34] #  [1, 17, 128, 128] 注意这里只取了最后stage的tag输出
            }

        tmp1 = array2dict(func([inp])) # 进行网络推理
        tmp2 = array2dict(func([inp[:,::-1]])) # 将图片左右镜像翻转之后再次预测
        # import matplotlib.pyplot as plt
        # plt.imshow(inp[:, ::-1])
        # plt.show()
        tmp = {}
        for ii in tmp1:
            tmp[ii] = np.concatenate((tmp1[ii], tmp2[ii]),axis=0) # det and tag [2,4, 17, 128, 128] [2, 17, 128, 128]
        # 将翻转之后的图像检测结果也结合起来,主要关节点的序号也需要利用flipref来变换一下, 这里只要最后得一个stage的输出
        det = tmp['det'][0, -1] + tmp['det'][1, -1, :, :, ::-1][flipRef] # [17, 128, 128]
        if det.max() > 10:
            continue
        if dets is None:
            dets = det # [17, 128, 128]
            mat = np.linalg.pinv(np.array(mat_).tolist() + [[0,0,1]])[:2] # 计算先前图像变换的逆矩阵
        else:
            dets = dets + resize(det, dets.shape[1:3]) 

        if abs(i-1)<0.5: # 将tags预测与resize到与det大小一致
            res = dets.shape[1:3] # 已有的检测结果对应的像素大小 要关节点的序号也需要利用flipref来变换一下
            tags += [resize(tmp['tag'][0], res), resize(tmp['tag'][1,:, :, ::-1][flipRef], res)]

    if dets is None or len(tags) == 0:
        return [], []

    tags = np.concatenate([i[:,:,:,None] for i in tags], axis=3) # [17, 128, 128, 2] 拼接起来
    dets = dets/len(scales)/2 # [17, 128, 128] #将不同尺度的检测结果平均
    
    dets = np.minimum(dets, 1) # [17 128 128]
    grouped = parser.parse(np.float32([dets]), np.float32([tags]))[0] # [num_person, 17, 5]
    # 进行group操作得到每个检测个体

    scores = [i[:, 2].mean() for  i in grouped] # [num_person] 用heatmap计算分数the score for every instance

    for i in range(len(grouped)): # 对于检测出来的每一个人
        grouped[i] = refine(dets, tags, grouped[i]) # 尝试找到那些没有检测到的点
    #grouped [N, 17, 5]
    if len(grouped) > 0:
        grouped[:,:,:2] = kpt_affine(grouped[:,:,:2] * 4, mat) # 把检测结果投到原图上面去 4是原图的下采样 512--》 128
    return grouped, scores

def coco_eval(prefix, dt, gt):
    """
    Evaluate the result with COCO API
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    for _, i in enumerate(sum(dt, [])):
        i['id'] = _+1

    image_ids = []
    import copy
    gt = copy.deepcopy(gt)

    dic = pickle.load(open(valid_filepath, 'rb'))
    paths, anns, idxes, info = [dic[i] for i in ['path', 'anns', 'idxes', 'info']]

    widths = {}
    heights = {}
    for idx, (a, b) in enumerate(zip(gt, dt)):
        if len(a)>0:
            for i in b:
                i['image_id'] = a[0]['image_id']
            image_ids.append(a[0]['image_id'])
        if info[idx] is not None:
            widths[a[0]['image_id']] = info[idx]['width']
            heights[a[0]['image_id']] = info[idx]['height']
        else:
            widths[a[0]['image_id']] = 0
            heights[a[0]['image_id']] = 0
    image_ids = set(image_ids)

    import json
    cat = [{'supercategory': 'person', 'id': 1, 'name': 'person', 'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]], 'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']}]
    with open(prefix + '/gt.json', 'w') as f:
        json.dump({'annotations':sum(gt, []), 'images':[{'id':i, 'width': widths[i], 'height': heights[i]} for i in image_ids], 'categories':cat}, f)

    with open(prefix + '/dt.json', 'w') as f:
        json.dump(sum(dt, []), f)

    coco = COCO(prefix + '/gt.json')
    coco_dets = coco.loadRes(prefix + '/dt.json')
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
    coco_eval.params.imgIds = list(image_ids)
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

def genDtByPred(pred, image_id = 0):
    """
    Generate the json-style data for the output
    pred =[N, 17, 3]
    """
    ans = []
    for i in pred: # 对应每一个个体的检测结果
        val = pred[i] if type(pred) == dict else i  #[17, 3]
        if val[:, 2].max()>0: #分数要大于0
            tmp = {'image_id':int(image_id), "category_id": 1, "keypoints": [], "score":float(val[:, 2].mean())} # 输出格式
            p = val[val[:, 2]> 0][:, :2].mean(axis = 0) #所有有效关节点的重心位置
            for j in val: # 对于每一个关节点
                if j[2]>0.: #  如果大于0则用预测位置
                    tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
                else: # 如果分数不可靠就用重心位置
                    tmp["keypoints"] += [float(p[0]), float(p[1]), 1]
            ans.append(tmp)
    return ans

def get_img(inp_res = 512):
    """
    Load validation images
    """
    if os.path.exists(valid_filepath) is False:
        from utils.build_valid import main
        main()

    dic = pickle.load(open(valid_filepath, 'rb'))
    paths, anns, idxes, info = [dic[i] for i in ['path', 'anns', 'idxes', 'info']]

    total = len(paths)
    tr = tqdm.tqdm( range(0, total), total = total )
    for i in tr:
        img = cv2.imread(paths[i])[:,:,::-1]
        yield anns[i], img

def main():
    from train import init
    func, config = init() # 模型定义与装载模型
    mode = config['opt'].mode

    def runner(imgs):
        return func(0, config, 'inference', imgs=torch.Tensor(np.float32(imgs)))['preds']

    def do(img):
        ans, scores = multiperson(img, runner, mode) # [N, 17, 5] [N]
        if len(ans) > 0:
            ans = ans[:,:,:3]# [N, 17, 3] --x, y, value

        pred = genDtByPred(ans) # [N]  Generate the json-style data for the output

        for i, score in zip( pred, scores ):
            i['score'] = float(score)
        return pred # 图片的预测结果

    gts = []
    preds = []

    idx = 0
    for anns, img in get_img(inp_res=-1): # here return image without rescale
        idx += 1
        gts.append(anns) # 注意是multi person
        preds.append(do(img)) # 预测结果
        if True:
            img_tmp = img.copy()
            for i in preds[idx-1]: #对于这个检测结果的每一个个体
                draw_limbs(img_tmp, i['keypoints'])
            plt.imshow(img_tmp)
            plt.show()
            cv2.imwrite('{}.jpg'.format(idx), img_tmp[:,:,::-1])

    prefix = os.path.join('exp', config['opt'].exp)
    coco_eval(prefix, preds, gts)

if __name__ == '__main__':
    main()
