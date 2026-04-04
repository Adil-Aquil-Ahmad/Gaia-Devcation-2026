import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OffroadDataset(Dataset):
    def __init__(self, images_dir, masks_dir, class_mapping, transforms=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.class_mapping = class_mapping
        
        self.images = sorted(os.listdir(images_dir))
        # Ensure only images are loaded
        self.images = [f for f in self.images if f.endswith('.png') or f.endswith('.jpg')]
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        # Since class IDs go up to 10000, we read with IMREAD_UNCHANGED to keep 16-bit data intact if applicable
        mask_path = os.path.join(self.masks_dir, img_name)
        if not os.path.exists(mask_path):
            mask_path = mask_path.replace('.jpg', '.png')
            
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # If mask is missing or incorrectly formatted, return zeros (edge case handling)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.int32)
            
        # If mask happens to be RGB, we typically take the first channel if IDs are just stored across or it needs special decoding.
        # But per standard semantic datasets, it will be 1 channel containing pixel mapping.
        if len(mask.shape) == 3:
            mask = mask[:, :, 0] # Assuming grayscale-like values stored as 3 channels
            
        # Remap non-contiguous values (100, 200, 300, 7100, 10000) to 0-9
        remapped_mask = np.zeros_like(mask, dtype=np.int64)
        for original_id, new_id in self.class_mapping.items():
            remapped_mask[mask == original_id] = new_id
            
        if self.transforms:
            augmented = self.transforms(image=image, mask=remapped_mask)
            image = augmented['image']
            remapped_mask = augmented['mask']
            
        return image, remapped_mask.long(), img_name

def get_train_transforms(img_height, img_width):
    # Strong augmentation pipeline per user requirements
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2), # In desert off-road, vertical flip might be rare but good for noise resistance
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.PadIfNeeded(min_height=img_height, min_width=img_width, border_mode=cv2.BORDER_CONSTANT),
        A.RandomCrop(height=img_height, width=img_width),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms(img_height, img_width):
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
