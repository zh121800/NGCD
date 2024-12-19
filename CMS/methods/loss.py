from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count   # n_views
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask        

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class ConMeanShiftLoss(nn.Module):
    def __init__(self, args):
        # 初始化ConMeanShiftLoss类
        super(ConMeanShiftLoss, self).__init__()
        
        # 设置每个样本的视角数（n_views），即每个图像有多少不同的增强视角
        self.n_views = args.n_views
        
        # alpha是控制特征和KNN嵌入之间加权的超参数
        self.alpha = args.alpha
        
        # temperature是控制对比损失温度的超参数，用于调整 logits 的尺度
        self.temperature = args.temperature
        
        # 选择计算设备，CUDA（GPU）优先，如果没有则使用CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, knn_emb, features):        
        # 计算每个批次的样本数量（每个视角的样本数）
        batch_size = int(features.size(0)) // self.n_views

        # 创建一个单位矩阵，表示正样本对的掩码（每个样本与自己是正样本）
        positive_mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        
        # 将单位矩阵重复n_views次，形成正样本对的掩码（每个视角都包含自身和其他视角的正样本）
        positive_mask = positive_mask.repeat(self.n_views, self.n_views)
        
        # 创建一个全为1的矩阵，表示负样本对的掩码
        negative_mask = torch.ones_like(positive_mask)
        
        # 将正样本掩码位置设为0，剩下的位置为负样本
        negative_mask[positive_mask > 0] = 0
        
        # 创建一个单位矩阵，防止计算样本与自己进行对比
        self_mask = torch.eye(batch_size * self.n_views, dtype=torch.float32).to(self.device)
        
        # 将自己与自己的对比掩码位置设为0，避免样本与自身对比
        positive_mask[self_mask > 0] = 0

        # 计算均值迁移后的特征向量（线性加权features和knn_emb）
        meanshift_feat = (1 - self.alpha) * features + self.alpha * knn_emb
        
        # 计算每个样本特征的L2范数并进行归一化，使得每个特征向量的模长为1
        norm = torch.sqrt(torch.sum((torch.pow(meanshift_feat, 2)), dim=-1)).unsqueeze(1).detach()
        meanshift_feat = meanshift_feat / norm
        
        # 计算均值迁移后的特征之间的点积并除以温度（temperature），得到 logits,温度是为了避免logits过大或者过小。
        anchor_dot_contrast = torch.div(torch.matmul(meanshift_feat, meanshift_feat.T), self.temperature)

        # 为了数值稳定性，从每行的最大值中减去logits
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 计算log_prob，通过对数归一化计算对比损失的对数概率
        exp_logits = torch.exp(logits) * negative_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))        

        # 计算正样本对的平均对数似然损失
        loss = - ((positive_mask * log_prob).sum(1) / positive_mask.sum(1))
        
        # 对所有视角和样本计算平均损失
        loss = loss.view(self.n_views, batch_size).mean()

        return loss  # 返回计算得到的损失

