import torch
import torch.nn as nn
import copy
import numpy as np

# ============================================================================
# METRICS AND LOSS FUNCTIONS
# (Standardized on Task 2 logic)
#
# ============================================================================

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Dice Coefficient - measures overlap (0-1, higher is better)
    Main metric for segmentation tasks
    """
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


def iou_score(pred, target, smooth=1e-6):
    """
    IoU (Intersection over Union) - alternative overlap metric (0-1, higher is better)
    """
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


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss
    Best for segmentation - handles class imbalance well
    """
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

# ============================================================================
# TRAINING AND VALIDATION LOOPS
# (Modified from Task 2 to remove tqdm)
#
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()  # Set model to training mode
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
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
        total_dice += dice_coefficient(outputs, masks)
        total_iou += iou_score(outputs, masks)
    
    # Return average metrics for the epoch
    return (total_loss / len(dataloader), 
            total_dice / len(dataloader), 
            total_iou / len(dataloader))


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():  # Don't calculate gradients
        # Iterate over dataloader without tqdm
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

# ============================================================================
# EARLY STOPPING CLASS
# (Adapted from Task 1, modified to MAXIMIZE Dice score)
#
# ============================================================================

class EarlyStopping:
    """Implements early stopping based on validation metric (e.g., Dice)."""
    
    def __init__(self, patience=5, verbose=True, delta=0, mode='max', 
                 path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            verbose (bool): If True, prints a message for each improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): 'min' for loss, 'max' for metrics like Dice/IoU.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

        if self.mode == 'min':
            self.val_metric_best = np.inf
        else:
            self.val_metric_best = -np.inf

    def __call__(self, val_metric, model):
        
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        
        elif (self.mode == 'max' and score < self.best_score + self.delta) or \
             (self.mode == 'min' and score > self.best_score - self.delta):
            # Metric did not improve
            self.counter += 1
            if self.verbose:
                print(f'  No improvement ({self.counter}/{self.patience})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Metric improved
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        """Saves model when validation metric improves."""
        # if self.verbose:
        #     if self.mode == 'max':
        #         print(f'SAVED best model (Dice: {val_metric:.4f})')
        #     else:
        #         print(f'SAVED best model (Loss: {val_metric:.4f})')
        # The print statement for saving the best model has been removed as requested.
        # Save a deep copy of the model's state_dict
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_metric_best = val_metric
        # Optional: save to disk
        # torch.save(model.state_dict(), self.path)