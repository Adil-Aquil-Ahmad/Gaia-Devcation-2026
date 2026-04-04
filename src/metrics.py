import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss

class HybridLoss(nn.Module):
    """
    User requests: Loss = 0.5 * CrossEntropyLoss + 0.5 * DiceLoss
    CrossEntropy improves pixel-wise classification.
    DiceLoss improves segmentation overlap and directly boosts IoU.
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(HybridLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(mode='multiclass')
        
    def forward(self, logits, targets):
        # logits shape: (B, C, H, W)
        # targets shape: (B, H, W)
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice

def compute_iou(preds, targets, num_classes):
    """
    Compute Intersection over Union (IoU) per class and mean IoU.
    preds: (B, H, W) integer targets
    targets: (B, H, W) integer targets
    """
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Ignore index 255 if using ignore_index, but we don't have it defined right now.
    
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in mean IoU
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            
    # Calculate valid mean IoU discarding NaNs
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    
    return ious, mean_iou

import numpy as np
