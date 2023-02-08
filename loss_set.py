# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 20:59:25 2021

@author: wangm
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmax(nn.Module):

    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.3,
                 s=15):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-9)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-9)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        delt_costh = torch.zeros_like(costh).scatter_(1, lb.unsqueeze(1), self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss
    
    
# version 1: use torch.autograd
class LargeMarginSoftmaxV1(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV1, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam
        self.ce_crit = nn.CrossEntropyLoss(
                reduction='none', ignore_index=ignore_index)


    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float()
        # logits.retain_grad()
        # logits.register_hook(lambda grad: grad)
        with torch.no_grad():
            num_classes = logits.size(1)
            coeff = 1. / (num_classes - 1.)
            lb = label.clone().detach()
            mask = label == self.ignore_index
            lb[mask] = 0
            idx = torch.zeros_like(logits).scatter_(1, lb.unsqueeze(1), 1.)

        lgts = logits - idx * 1.e6
        q = lgts.softmax(dim=1)
        q = q * (1. - idx)

        log_q = lgts.log_softmax(dim=1)
        log_q = log_q * (1. - idx)
        mg_loss = ((q - coeff) * log_q) * (self.lam / 2)
        mg_loss = mg_loss * (1. - idx)
        mg_loss = mg_loss.sum(dim=1)

        ce_loss = self.ce_crit(logits, label)
        loss = ce_loss + mg_loss
        loss = loss[mask == 0]

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

criteria = LargeMarginSoftmaxV1()
logits = torch.randn(1, 4, 10) # nchw, float/half
lbs = torch.randint(0, 4, (1, 10)) # nhw, int64_t
loss = criteria(logits, lbs)