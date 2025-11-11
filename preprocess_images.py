# preprocess_image.py
# This script will scale all images within the KvasirSEG dataset to all resolutions defined within RESOLUTIONS in the Config class, and then back up to the target size.
# The advantage of doing this is we don't need to process 8000 images every time we run the task 1 script.
#
# Candidate 27 and Candidate 16

import os
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class Config:
    # Original dataset paths
    DATASET_PATH = "data/kvasir-seg"    # Change this to match local dataset path
    IMAGE_DIR = "images"
    MASK_DIR = "masks"
    
    RESOLUTIONS = [512, 256, 128, 64]
    TARGET_SIZE = 256
    
    # New path for pre-processed data
    PREPROCESSED_PATH = "data/kvasir-seg-preprocessed"

config = Config()

base_path = Path(config.DATASET_PATH)
preprocessed_base = Path(config.PREPROCESSED_PATH)

image_paths = sorted(list((base_path / config.IMAGE_DIR).glob('*.jpg')))
mask_paths = sorted(list((base_path / config.MASK_DIR).glob('*.jpg')))

print(f"Found {len(image_paths)} images and {len(mask_paths)} masks.")

# Loop over each resolution
for res in config.RESOLUTIONS:
    print(f"--- Processing for resolution {res}x{res} ---")
    
    # Create output directories
    img_out_dir = preprocessed_base / str(res) / config.IMAGE_DIR
    mask_out_dir = preprocessed_base / str(res) / config.MASK_DIR
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    for img_path in tqdm(image_paths, desc=f"Images ({res}px)"):
        try:
            image = Image.open(img_path).convert('RGB')

            # Scale down to resolution
            image = TF.resize(image, (res, res), interpolation=Image.BILINEAR)

            # Scale back to target size
            image = TF.resize(image, (config.TARGET_SIZE, config.TARGET_SIZE), interpolation=Image.BILINEAR)
            
            # Save to new location
            image.save(img_out_dir / img_path.name)
        except Exception as e:
            print(f"Error processing image {img_path.name}: {e}")
            
    # Process masks
    for mask_path in tqdm(mask_paths, desc=f"Masks ({res}px)"):
        try:
            mask = Image.open(mask_path).convert('L')

            # Scale down to resolution
            mask = TF.resize(mask, (res, res), interpolation=Image.NEAREST)
            
            # Scale back to target size
            mask = TF.resize(mask, (config.TARGET_SIZE, config.TARGET_SIZE), interpolation=Image.NEAREST)
            
            # Save to new location
            mask.save(mask_out_dir / mask_path.name)
        except Exception as e:
            print(f"Error processing mask {mask_path.name}: {e}")

print("Pre-processing complete!")