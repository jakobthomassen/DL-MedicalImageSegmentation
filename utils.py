# utils.py
# This script holds metrics, loss functions and training + validation loops. 
# Storing them here and reusing them in different notebooks streamlines the process and keeps the methods uniform in both tasks.
# Candidate 27 and Candidate 16

import torch
import torch.nn as nn
import copy
import numpy as np

# METRICS AND LOSS FUNCTIONS

def dice_coefficient(pred, target, smooth=1e-6): # Dice, measures overlap (0-1, higher is better)

    # Apply sigmoid to logits and threshold to get binary predictions
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    # Flatten tensors for calculation
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def iou_score(pred, target, smooth=1e-6): # Intersection over Union (0-1, higher is better)

    # Apply sigmoid to logits and threshold
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


class DiceBCELoss(nn.Module): # Combined Dice + BCE Loss. Best for segmentation

    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # BCE loss
        bce_loss = self.bce(pred, target)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-6
        
        # Flatten for Dice calculation
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
        
        # Combine losses
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss
    
    
# TRAINING AND VALIDATION LOOPS

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()  # Set model to training mode
    total_loss = 0
    
    # Iterate over dataloader without tqdm
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Get predictions
        
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()  # Calculate gradients
        optimizer.step()  # Update weights
        
        # Track metrics
        total_loss += loss.item()
    
    # Return average metrics for the epoch
    return (total_loss / len(dataloader))

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():  # Don't calculate gradients
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks)
            total_iou += iou_score(outputs, masks)
    
    # Return average metrics for the epoch
    return (total_loss / len(dataloader), 
            total_dice / len(dataloader), 
            total_iou / len(dataloader))



class EarlyStopping:
    
    def __init__(self, patience=5, verbose=True, delta=0, mode='max', 
                 path='checkpoint.pth'):

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
        # Add holders for all best metrics
        self.best_loss = np.inf
        self.best_iou = -np.inf

        if self.mode == 'min':
            self.val_metric_best = np.inf
        else:
            self.val_metric_best = -np.inf

    def __call__(self, val_metric, val_loss, val_iou, model):
        """
        Updated call signature to accept all relevant metrics
        """
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, val_loss, val_iou, model)
        
        elif (self.mode == 'max' and score < self.best_score + self.delta) or \
             (self.mode == 'min' and score > self.best_score - self.delta):
            # Metric did not improve
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Metric improved
            self.best_score = score
            self.save_checkpoint(val_metric, val_loss, val_iou, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, val_loss, val_iou, model):
        # Save a copy of the model's state_dict
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_metric_best = val_metric
        
        # Save the other associated metrics
        self.best_loss = val_loss
        self.best_iou = val_iou