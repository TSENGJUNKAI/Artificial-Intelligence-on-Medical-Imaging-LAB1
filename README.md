# Artificial-Intelligence-on-Medical-Imaging-LAB1
Detect Pneumonia from chest X-Ray images
## Project Overview
This project implements pneumonia classification using deep learning models (ResNet18 and ResNet50) on chest X-ray images.

## Dataset
- **Source**: [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: NORMAL, PNEUMONIA
- **Split**: Train / Validation / Test

## Requirements
torch/torchvision/numpy/matplotlib/seaborn/scikit-learn/Pillow

## Installation
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn Pillow
```
## Project Structure
```bash
.
├── train.py              # Training script
├── inference.py          # Inference script
├── README.md            # This file
├── chest_xray/          # Dataset folder
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
```
## Training
```bash
python train.py
```
This will:
Train ResNet18 and ResNet50 models
Save models as ResNet18_model.pth and ResNet50_model.pth
Generate training curves and confusion matrices

## Inference
```bash
python inference.py
```
This will:
Load trained models
Evaluate on test dataset
Generate confusion matrices

## Results
ResNet18: Test Accuracy: XX.XX%, F1-Score: X.XX
ResNet50: Test Accuracy: XX.XX%, F1-Score: X.XX

## Models
ResNet18: 18-layer Residual Network with transfer learning
ResNet50: 50-layer Residual Network with transfer learning

## Data Augmentation
Resize to 224x224
Random Horizontal Flip
Random Rotation (±10°)
Color Jitter (brightness, contrast)
Normalization (ImageNet mean and std)

## Author
[chunkai] - [314113012]

License
This project is for educational purposes only.
