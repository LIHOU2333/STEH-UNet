import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

# __all__ = ['BCEDiceLoss', 'LovaszHingeLoss']
__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'DiceLoss', 'BCELoss', 'BCEWithLogitsLoss']

# Dice Loss  正常
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        intersection = (input * target).sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (input.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smooth)
        return 1 - dice.mean()


# Binary Cross-Entropy (BCE) Loss   报错
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input, target):
        return F.binary_cross_entropy(input, target)

# Binary Cross-Entropy with Logits Loss  指标为0
class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target)


# 二元交叉熵（Binary Cross-Entropy, BCE）损失和Dice系数损失     正常
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
