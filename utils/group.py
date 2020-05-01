# Functions for grouping tags
import numpy as np
from munkres import Munkres
import torch

def py_max_match(scores):  # 匈牙利算法求解二分图匹配问题
    m = Munkres()
    tmp = m.compute(-scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp

class Params:
    def __init__(self):
        self.num_parts = 17
        self.detection_threshold = 0.2
        self.tag_threshold = 1.
        self.partOrder = [i-1 for i in [1,2,3,4,5,6,7,12,13,8,9,10,11,14,15,16,17]]
        self.max_num_people = 30
        self.use_detection_val = 0
        self.ignore_too_much = False

def match_by_tag(inp, params, pad=False):
    tag_k, loc_k, val_k = inp # 极值点的tag 位置 与heatmap value [17, 30, 2] [17, 30, 2] [17, 30]
    assert type(params) is Params
    default_ = np.zeros((params.num_parts, 3 + tag_k.shape[2])) # [17, 5]

    dic = {}
    dic2 = {}
    Flag = False
    if Flag:
        # show the tags
        m = val_k > params.detection_threshold
        tag_m = tag_k[m]



    # 用不同的embedding来表示一个单独的个体
    for i in range(params.num_parts): # 对于每一个part
        ptIdx = params.partOrder[i]

        tags = tag_k[ptIdx] # 取出对应part的信息 [30, 2]
        joints = np.concatenate((loc_k[ptIdx], val_k[ptIdx, :, None], tags), 1) # [30, 5]
        mask = joints[:, 2] > params.detection_threshold # 检测分数需要超过阈值
        tags = tags[mask] # [N, 2]
        joints = joints[mask] # [N, 5]
        if i == 0 or len(dic) == 0: #  初始化检测结果
            for tag, joint in zip(tags, joints): # [2] [5]  如果是第一个关节点， 就先用每一个tag初始化每一个个体
                dic.setdefault(tag[0], np.copy(default_))[ptIdx] = joint # 用tag[0]作为每个检测个体的reference embedding，并有已有的part检测结果
                dic2[tag[0]] = [tag] # 作为每个检测个体的tags累计
        else:
            actualTags = list(dic.keys())[:params.max_num_people] # 已有的个体的reference embedding
            actualTags_key = actualTags # [N1]
            actualTags = [np.mean(dic2[i], axis = 0) for i in actualTags] # 得到每一个个体的embedding均值 （沿着关节点维度）

            if params.ignore_too_much and len(actualTags) == params.max_num_people:
                continue
            diff = ((joints[:, None, 3:] - np.array(actualTags)[None, :, :])**2).mean(axis = 2) ** 0.5 # 计算当前part的tags与已有个体的embedding插值
            if diff.shape[0]==0: # [N ,N1] N是当前part的检测个数 N1是已经检测出来的人体个数
                continue

            diff2 = np.copy(diff)

            if params.use_detection_val :
                diff = np.round(diff) * 100 - joints[:, 2:3]

            if diff.shape[0]>diff.shape[1]: # 当前part检测数目比已有的人数还要多
                diff = np.concatenate((diff, np.zeros((diff.shape[0], diff.shape[0] - diff.shape[1])) + 1e10), axis = 1)

            pairs = py_max_match(-diff) # [N, 2] #get minimal matching 求解最佳分配
            for row, col in pairs: # 对于ptIdx part的N个检测
                if row<diff2.shape[0] and col < diff2.shape[1] and diff2[row][col] < params.tag_threshold: # 匹配的点的tag差异不能太大
                    dic[actualTags_key[col]][ptIdx] = joints[row] # 添加到已有检测人体的ptIdx关节点的检测结果
                    dic2[actualTags_key[col]].append(tags[row]) # tag部分也添加进去
                else: # 检测出新的人
                    key = tags[row][0] # 用第一个tag作为 新的reference embedding
                    dic.setdefault(key, np.copy(default_))[ptIdx] = joints[row] # 检测结果
                    dic2[key] = [tags[row]]

    ans = np.array([dic[i] for i in dic]) # [N, 17, 5]  N为检测出来的人个数
    if pad:
        num = len(ans)
        if num < params.max_num_people:
            padding = np.zeros((params.max_num_people-num, params.num_parts, default_.shape[1]))
            if num>0: ans = np.concatenate((ans, padding), axis = 0)
            else: ans = padding
        return np.array(ans[:params.max_num_people]).astype(np.float32)
    else:
        return np.array(ans).astype(np.float32) # [N, 17, 5]  N为检测出来的人个数

class HeatmapParser():
    def __init__(self, detection_val=0.03, tag_val=1.):
        from torch import nn
        self.pool = nn.MaxPool2d(3, 1, 1)
        param = Params()
        param.detection_threshold = detection_val
        param.tag_threshold = tag_val
        param.ignore_too_much = True
        param.max_num_people = 30
        param.use_detection_val = True
        self.param = param

    def nms(self, det):
        # suppose det is a tensor
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float() # 通过这个找到极值点
        det = det * maxm
        return det # 非极值点都置为0
        '''
        #codes for test
        import matplotlib.pyplot as plt
        plt.imshow(det[0,0,:,:].numpy())
        plt.show()
        plt.imshow(self.pool(det)[0,0,:,:].numpy())
        plt.show()
        plt.imshow(torch.eq(self.pool(det), det).float()[0,0,:,:].numpy())
        plt.show()
        '''

    def match(self, tag_k, loc_k, val_k): # [1, 17, 30, 2] [1, 17, 30, 2] [1, 17, 30]
        match = lambda x:match_by_tag(x, self.param) # 对于每一个图片 进行match操作，组成每一个人检测
        return list(map(match, zip(tag_k, loc_k, val_k)))

    def calc(self, det, tag):
        with torch.no_grad():
            det = torch.Tensor(det) # [bs, 17, 128, 128] 实际上bs=1
            tag =torch.Tensor(tag) # [bs, 17, 128, 128, 2]

            det = self.nms(det) # 对热图做nms
            h = det.size()[2]
            w = det.size()[3]
            det = det.view(det.size()[0], det.size()[1], -1)
            tag = tag.view(tag.size()[0], tag.size()[1], det.size()[2], -1) #(1, 17, 128*128, 2)
            # ind (1, 17, 30)
            # val (1, 17, 128*128)
            # tag (1, 17, 128*128, -1)
            val_k, ind = det.topk(self.param.max_num_people, dim=2) # top_k的极值点
            tag_k = torch.stack([torch.gather(tag[:,:,:,i], 2, ind) for i in range(tag.size()[3])], dim=3)
            # 这些极值点对应的tag值 [1, 17, 30, 2]
            x = ind % w # 求余数得到极值点的x坐标 [1. 17. 30]
            y = (ind / w).long() # 极值点的y坐标 [1, 17, 30]
            ind_k = torch.stack((x, y), dim=3) # [1, 17, 30, 2]
            ans = {'tag_k': tag_k, 'loc_k': ind_k, 'val_k': val_k} # 返回极值点的tag 位置 与heatmap value
            return {key:ans[key].cpu().data.numpy() for key in ans}

    def adjust(self, ans, det): # 检测出来的N个人([N, 17, 5]) 检测出来的每个热图 [1, 17, 128, 128]
        for batch_id, people in enumerate(ans): # 分图片处理 一般就是一个图片
            for people_id, i in enumerate(people): # 对于检测出来的每个人 关节信息 i [17, 5]
                for joint_id, joint in enumerate(i): # 对于检测出来的某一个关节点joint [5]
                    if joint[2]>0: # i如果heatmap大于0 说明这个关节点被检测出来了
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        #print(batch_id, joint_id, det[batch_id].shape)
                        tmp = det[batch_id][joint_id] #这个关节的热图 [128, 128]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
                            y+=0.25
                        else:
                            y-=0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
                            x+=0.25
                        else:
                            x-=0.25
                        ans[batch_id][people_id, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    def parse(self, det, tag, adjust=True):
        ans = self.match(**self.calc(det, tag)) # ([N, 17, 5])calc进行匹配 calc得到极值点的tag 位置 与heatmap value match：匹配成每一个检测人个体
        if adjust:
            ans = self.adjust(ans, det) # 进行关节点位置的精调
        return ans # ([N, 17, 5])
