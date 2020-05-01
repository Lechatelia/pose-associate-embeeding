"""
@Time:     2020/04/30 17:35
@Author:   Jinguo Zhu
@Email:    lechatelia@stu.xjtu.edu.cn
@File:     util.py
@Software: PyCharm

@description:

"""


import cv2
import scipy.misc
import numpy as np

flipRef = [i-1 for i in [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16] ]

part_labels = ['nose','eye_l','eye_r','ear_l','ear_r',
               'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
               'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']
part_idx = {b:a for a, b in enumerate(part_labels)}

def draw_limbs(inp, pred):
    def link(a, b, color):
        if part_idx[a] < pred.shape[0] and part_idx[b] < pred.shape[0]:
            a = pred[part_idx[a]]
            b = pred[part_idx[b]]
            if a[2]>0.07 and b[2]>0.07:
                cv2.line(inp, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 6)

    pred = np.array(pred).reshape(-1, 3)
    bbox = pred[pred[:,2]>0]
    a, b, c, d = bbox[:,0].min(), bbox[:,1].min(), bbox[:,0].max(), bbox[:,1].max()

    cv2.rectangle(inp, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 2)

    link('nose', 'eye_l', (255, 0, 0))
    link('eye_l', 'eye_r', (255, 0, 0))
    link('eye_r', 'nose', (255, 0, 0))

    link('eye_l', 'ear_l', (255, 0, 0))
    link('eye_r', 'ear_r', (255, 0, 0))

    link('ear_l', 'sho_l', (255, 0, 0))
    link('ear_r', 'sho_r', (255, 0, 0))
    link('sho_l', 'sho_r', (255, 0, 0))
    link('sho_l', 'hip_l', (0, 255, 0))
    link('sho_r', 'hip_r',(0, 255, 0))
    link('hip_l', 'hip_r', (0, 255, 0))

    link('sho_l', 'elb_l', (0, 0, 255))
    link('elb_l', 'wri_l', (0, 0, 255))

    link('sho_r', 'elb_r', (0, 0, 255))
    link('elb_r', 'wri_r', (0, 0, 255))

    link('hip_l', 'kne_l', (255, 255, 0))
    link('kne_l', 'ank_l', (255, 255, 0))

    link('hip_r', 'kne_r', (255, 255, 0))
    link('kne_r', 'ank_r', (255, 255, 0))







def adjustKeypoint(tmp, loc):
    ans = []
    for x, y in zip(*loc):
        xx, yy = x, y
        if tmp[xx, min(yy+1, tmp.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
            y+=0.25
        else:
            y-=0.25

        if tmp[min(xx+1, tmp.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
            x+=0.25
        else:
            x-=0.25
        ans.append((x + 0.5, y + 0.5))
    return np.array(ans)


def resize(*args):
    im = args[0]
    if im.ndim == 3 and im.shape[2] > 3:
        res = args[1]
        new_im = np.zeros((res[0], res[1], im.shape[2]), np.float32)
        for i in range(im.shape[2]):
            if im[:,:,i].max() > 0:
                new_im[:,:,i] = resize(im[:,:,i], res, *args[2:])
        return new_im
    else:
        return scipy.misc.imresize(*args).astype(np.float32) / 255

def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)

def crop(img, center, scale, res, rot=0):
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]

    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    tmp = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    #print(old_y[0],old_y[1], old_x[0],old_x[1])
    #print(new_y[0], new_y[1], new_x[0], new_x[1], tmp.max())
    if old_x[0]<old_x[1] and old_y[0] < old_y[1]:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = tmp

    if not rot == 0:
        # Remove padding
        # something is very stupid that it would convert 1. to 255 here
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    #print(img.max(), tmp.max(), new_img.max(), new_img.astype(np.uint8).max())
    return resize(new_img.astype(np.uint8), res)

def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot( np.concatenate((kpt, kpt[:, 0:1]*0+1), axis = 1), mat.T ).reshape(shape)