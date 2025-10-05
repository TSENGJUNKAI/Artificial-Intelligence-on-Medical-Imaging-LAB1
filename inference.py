#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import seaborn as sns

# 檢查 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==================== 定義資料轉換 ====================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== 自定義 Dataset ====================
class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        classes = ['NORMAL', 'PNEUMONIA']
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpeg', '.jpg', '.png')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(idx)
        
        print(f'Loaded {len(self.images)} images')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

# ==================== 載入測試資料 ====================
test_dir = './chest_xray/test'
test_dataset = PneumoniaDataset(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ==================== 載入模型並推論 ====================
def inference(model_name, model_path):
    print(f'\n========== Inference with {model_name} ==========')
    
    # 建立模型架構
    if model_name == 'ResNet18':
        model = models.resnet18(pretrained=False)
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=False)
    else:
        raise ValueError('Model not supported')
    
    # 修改最後一層
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # 載入權重
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # 推論
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 計算指標
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    print(f'Test Accuracy: {acc:.4f}')
    print(f'Test F1-Score: {f1:.4f}')
    
    # 顯示分類報告
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=['NORMAL', 'PNEUMONIA']))
    
    # 繪製混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'inference_confusion_matrix_{model_name}.png', dpi=150)
    plt.show()
    
    return acc, f1

# ==================== 執行推論 ====================
inference('ResNet18', 'ResNet18_model.pth')
inference('ResNet50', 'ResNet50_model.pth')

print('\nInference completed!')

# %%
