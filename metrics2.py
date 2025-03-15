import numpy as np
import torch
import torch.nn.functional as F


# 混淆矩阵计算了四个基本元素：真阳性（TP）、真阴性（TN）、假阳性（FP）和假阴性（FN）。这些元素用于后续指标的计算。
def confusion_matrix(output, target):
    output = torch.sigmoid(output).data.cpu().numpy()
    output_ = output > 0.5
    target_ = target.data.cpu().numpy()

    tp = (output_ == 1) & (target_ == 1)
    tn = (output_ == 0) & (target_ == 0)
    fp = (output_ == 1) & (target_ == 0)
    fn = (output_ == 0) & (target_ == 1)

    tp, tn, fp, fn = tp.sum(), tn.sum(), fp.sum(), fn.sum()

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)

    return tp, tn, fp, fn, precision, recall, f1_score, accuracy

#  交并比 (IoU) 和 Dice系数
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (output_.sum() + target_.sum() + smooth)

    return iou, dice


# 最后，indicators 函数将上述所有指标综合起来，并返回结果。
def indicators(output, target):
    tp, tn, fp, fn, precision, recall, f1_score, accuracy = confusion_matrix(output, target)
    iou, dice = iou_score(output, target)

    return iou, dice, tp, tn, fp, fn, precision, recall, f1_score, accuracy
