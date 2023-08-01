# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import morphology
import numpy as np


def _one_hot(arr):
    arr = torch.unsqueeze(arr, 1)
    arr_neg = arr * 2
    arr_neg[arr_neg == 0] = 1
    arr_neg[arr_neg == 2] = 0
    arr_tmp = torch.cat((arr_neg, arr), dim=1)
    return arr_tmp

def _flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class DiceLoss(nn.Module):
    def __init__(self, normalization='sigmoid', weight=None):
        super(DiceLoss, self).__init__()
        assert normalization in ['sigmoid', 'softmax', 'none']
        self.normalization = normalization
        if normalization == 'sigmoid':
            self.norm = torch.sigmoid
        elif normalization == 'softmax':
            self.norm = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x
        self.weight = weight

    def forward(self, pred, target):
        smooth = 1e-6
        # pred = pred.squeeze(dim=1)
        if self.normalization == 'softmax':
            pred = self.norm(pred)
            target = _one_hot(target)
        elif self.normalization == 'sigmoid':
            pred = self.norm(pred)
            #target = torch.unsqueeze(target, 1)
        pred = _flatten(pred)
        target = _flatten(target)
        target = target.float()

        # compute per channel Dice Coefficient
        intersect = (pred * target).sum(-1)
        if self.weight is not None:
            intersect = self.weight * intersect

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (pred * pred).sum(-1) + (target * target).sum(-1)
        dice = (2 * intersect + smooth) / (denominator + smooth)
        return 1. - torch.mean(dice)

class CLDiceLoss(nn.Module):
    def __init__(self, normalization='sigmoid', weight=None):
        super(CLDiceLoss, self).__init__()
        assert normalization in ['sigmoid', 'softmax', 'none']
        self.normalization = normalization
        if normalization == 'sigmoid':
            self.norm = torch.sigmoid
        elif normalization == 'softmax':
            self.norm = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x
        self.weight = weight

    def forward(self, pred, pred1, target, target1):
        smooth = 1e-6
        # pred = pred.squeeze(dim=1)
        if self.normalization == 'softmax':
            pred = self.norm(pred)
            target = _one_hot(target)
        elif self.normalization == 'sigmoid':
            pred = self.norm(pred)
            pred1 = self.norm(pred1)
            #target = torch.unsqueeze(target, 1)
        pred = _flatten(pred)
        pred1 = _flatten(pred1)
        target = _flatten(target)
        target = target.float()
        target1 = _flatten(target1)
        target1 = target1.float()

        tprec = ((target * pred1).sum(-1) + smooth) / (pred1.sum(-1) + smooth)
        tsens = ((pred * target1).sum(-1) + smooth) / (target1.sum(-1) + smooth)
        cldice = (2 * tprec * tsens) / (tprec + tsens)
        return 1. - torch.mean(cldice)


class SoftCLDiceLoss(nn.Module):
    def __init__(self, normalization='sigmoid', weight=None):
        super(SoftCLDiceLoss, self).__init__()
        assert normalization in ['sigmoid', 'softmax', 'none']
        self.normalization = normalization
        if normalization == 'sigmoid':
            self.norm = torch.sigmoid
        elif normalization == 'softmax':
            self.norm = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x
        self.weight = weight

    def forward(self, pred, target, target1):
        smooth = 1e-6
        # pred = pred.squeeze(dim=1)
        if self.normalization == 'softmax':
            pred = self.norm(pred)
            target = _one_hot(target)
        elif self.normalization == 'sigmoid':
            pred = self.norm(pred)
            # target = torch.unsqueeze(target, 1)
        pred_f = _flatten(pred)
        pred_skel = _flatten(soft_skel(pred))
        target = _flatten(target)
        target = target.float()
        target_skel = _flatten(target1)
        target_skel = target_skel.float()

        tprec = ((target * pred_skel).sum(-1) + smooth) / (pred_skel.sum(-1) + smooth)
        tsens = ((pred_f * target_skel).sum(-1) + smooth) / (target_skel.sum(-1) + smooth)
        cldice = (2 * tprec * tsens) / (tprec + tsens)
        return 1. - torch.mean(cldice)

def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)

def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_=5):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


