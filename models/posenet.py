import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool
from extensions.AE.AE_loss import AEloss
from task.loss import HeatmapLoss

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, **kwargs):
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            Pool(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )
        self.features = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
            Conv(inp_dim, inp_dim, 3, bn=False),
            Conv(inp_dim, inp_dim, 3, bn=False)
        ) for i in range(nstack)] ) # hourglass 结构提取特征

        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] ) # 预测
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] ) # 用于融合特征和结构以便于多层次预测
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )

        self.nstack = nstack
        self.myAEloss = AEloss()
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2) # [bs, 3, 128, 128]
        x = self.pre(x) # [bs, 256, 128, 128]
        preds = []
        for i in range(self.nstack): # 4
            feature = self.features[i](x) # hourglass 提取特征 [bs, 256, 128, 128]
            preds.append( self.outs[i](feature) ) # add [bs, 68, 128, 128]
            if i != self.nstack - 1: # 特征融合 增强特征 作为下一个的输入 大小还是# [bs, 256, 128, 128]
                x = x + self.merge_preds[i](preds[-1]) + self.merge_features[i](feature) #作为下一个stage的输出
        return torch.stack(preds, 1) # [bs, 4, 68, 128, 128] # 4是因为有一个堆叠的网络输出

    def calc_loss(self, preds, keypoints=None, heatmaps = None, masks = None):
        dets = preds[:,:,:17]
        tags = preds[:,:,17:34]

        keypoints = keypoints.cpu().long()
        batchsize = tags.size()[0]

        tag_loss = []
        for i in range(self.nstack):
            tag = tags[:,i].contiguous().view(batchsize, -1, 1)
            tag_loss.append( self.myAEloss(tag, keypoints) ) # ae loss
        tag_loss = torch.stack(tag_loss, dim = 1).cuda(tags.get_device())

        detection_loss = []
        for i in range(self.nstack):
            detection_loss.append( self.heatmapLoss(dets[:,i], heatmaps, masks) )
        detection_loss = torch.stack(detection_loss, dim=1)
        return tag_loss[:,:,0], tag_loss[:,:,1], detection_loss
