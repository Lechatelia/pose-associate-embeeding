import torch
from torch import nn


class AEloss(nn.Module):
    def forward(self, tags, keypoints):
        """
        Inputs:
            tags: [batch_size, 17 * output_res * output_res, tag_dim] [24, 278528, 1]
            keypoints: [batch_size, max_num_people, 17, 2] [24, 30, 17, 2]
        Return:
            output: [batch_size, 2]
                output[:, 0]: push loss
                output[:, 1]: pull loss
        """
        pushes, pulls = [], []
        batch_size = tags.size(0)
        for i in range(batch_size): # 对于每一个image
            push, pull = self.single_image_tag_loss(tags[i], keypoints[i])
            pushes.append(push.reshape(1)) # [1]
            pulls.append(pull.reshape(1)) # [1]
        return torch.cat([torch.stack(pushes), torch.stack(pulls)], dim=1) # [bs, 2]

    def single_image_tag_loss(self, tags, keypoints):
        """
        Inputs:
            tag: [17 * output_res * output_res, tag_dim] [278528, 1]
            keypoints: [max_num_people, 17, 2] [30, 17, 2] keypints 的位置跟tag一样，进行了flatten处理
        """
        eps = 1e-6
        pull = 0
        tag_dim = tags.size(1) # tag维度
        mean_tags = []
        for keypoints_person in keypoints: #[17, 2] 对于其中一个人
            mask = keypoints_person[:, 1] > 0
            tags_person = tags[keypoints_person[mask][:, 0].long()] # [N, 1]
            if tags_person.size(0) == 0:
                continue
            mean_tags.append(torch.mean(tags_person, dim=0)) #均值
            pull += torch.mean(torch.pow(tags_person - mean_tags[- 1], 2).sum(1)) # pull loss

        if len(mean_tags) == 0:
            return torch.zeros([1]).cuda(), torch.zeros([1]).cuda()
            # i think it should be
            # return torch.zeros([]).cuda(), torch.zeros([]).cuda()

        mean_tags = torch.stack(mean_tags)  # [person_num, tag_dim]
        person_num = mean_tags.size(0)

        x = mean_tags.unsqueeze(1).expand(person_num, person_num, tag_dim)
        diff = torch.pow(x - x.permute(1, 0, 2), 2).sum(2)  # [person_num, person_num]
        upper_triangle_idx = diff.triu(1).nonzero() # 上三角元素
        diff = diff[upper_triangle_idx[:, 0], upper_triangle_idx[:, 1]]
        push = torch.exp(-diff).sum()
        return push / ((person_num - 1) * person_num + eps), pull / (person_num + eps)