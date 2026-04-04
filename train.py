import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import OffroadDataset, get_train_transforms, get_val_transforms
from src.model import create_model
from src.metrics import HybridLoss, compute_iou
from src.utils import load_config, setup_directories

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')
    return parser.parse_args()

def main():
    args = get_args()
    config = load_config(args.config)
    setup_directories(config['logging'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Setup data
    train_transforms = get_train_transforms(config['dataset']['img_height'], config['dataset']['img_width'])
    val_transforms = get_val_transforms(config['dataset']['img_height'], config['dataset']['img_width'])
    
    train_dataset = OffroadDataset(
        os.path.join(config['dataset']['train_dir'], 'Color_Images'),
        os.path.join(config['dataset']['train_dir'], 'Segmentation'),
        config['class_mapping'],
        transforms=train_transforms
    )
    
    val_dataset = OffroadDataset(
        os.path.join(config['dataset']['val_dir'], 'Color_Images'),
        os.path.join(config['dataset']['val_dir'], 'Segmentation'),
        config['class_mapping'],
        transforms=val_transforms
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                              shuffle=True, num_workers=config['training']['num_workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], 
                            shuffle=False, num_workers=config['training']['num_workers'])
    
    # Model
    model = create_model(
        arch=config['model']['architecture'],
        backbone=config['model']['backbone'],
        weights=config['model']['weights'],
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    # Loss, Optimizer, and Scaler for Mixed Precision
    criterion = HybridLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), 
                            lr=config['training']['learning_rate'], 
                            weight_decay=config['training']['weight_decay'])
    scaler = torch.amp.GradScaler(device='cuda')
    
    # Logging histories
    history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
    best_val_miou = 0.0
    
    # Training Loop
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks, _ in pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        all_val_ious = []
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, masks, _ in pbar_val:
                images, masks = images.to(device), masks.to(device)
                
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                
                for b in range(images.size(0)):
                    _, m_iou = compute_iou(preds[b:b+1], masks[b:b+1], config['model']['num_classes'])
                    all_val_ious.append(m_iou)
                    
        val_loss /= len(val_loader)
        epoch_val_miou = np.nanmean(all_val_ious)
        
        history['val_loss'].append(val_loss)
        history['val_miou'].append(epoch_val_miou)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {epoch_val_miou:.4f}")
        
        # Save best model
        if epoch_val_miou > best_val_miou:
            print(f"--> Valid mIoU improved from {best_val_miou:.4f} to {epoch_val_miou:.4f}. Saving model.")
            best_val_miou = epoch_val_miou
            save_path = os.path.join(config['logging']['checkpoint_dir'], 'best.pth')
            torch.save(model.state_dict(), save_path)
            
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(config['logging']['run_dir'], 'loss_curve.png'))
    plt.close()

if __name__ == '__main__':
    main()
