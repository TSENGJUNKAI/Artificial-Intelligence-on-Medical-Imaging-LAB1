#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

# 設定隨機種子
torch.manual_seed(42)
np.random.seed(42)

# 檢查 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
#%%
# ==================== 步驟 1: 定義資料轉換 ====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#%%
# ==================== 步驟 2: 自定義 DataLoader ====================
class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, transform=None, indices=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 讀取 NORMAL 和 PNEUMONIA 資料夾
        classes = ['NORMAL', 'PNEUMONIA']
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpeg', '.jpg', '.png')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(idx)  # NORMAL=0, PNEUMONIA=1
        
        # 如果有指定 indices，只使用這些索引
        if indices is not None:
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
        
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
#%%
# ==================== 步驟 3: 載入資料 ====================
train_dir = './chest_xray/train'
test_dir = './chest_xray/test'

# 載入訓練資料
temp_dataset = PneumoniaDataset(train_dir, transform=None)

# 切分 train/val (80%/20%)
indices = list(range(len(temp_dataset)))
train_indices, val_indices = train_test_split(
    indices, 
    test_size=0.2,
    stratify=temp_dataset.labels,
    random_state=42
)

print(f'Train: {len(train_indices)} images')
print(f'Val: {len(val_indices)} images')

# 創建 dataset
train_dataset = PneumoniaDataset(train_dir, transform=train_transform, indices=train_indices)
val_dataset = PneumoniaDataset(train_dir, transform=test_transform, indices=val_indices)
test_dataset = PneumoniaDataset(test_dir, transform=test_transform)

# 創建 DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#%%
# ==================== 步驟 4: 定義訓練和評估函數 ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels
#%%
# ==================== 步驟 5: 訓練模型函數 ====================
def train_model(model_name, num_epochs=10):
    print(f'\n========== Training {model_name} ==========')
    
    # 建立模型
    if model_name == 'ResNet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError('Model not supported')
    
    # 修改最後一層
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 記錄訓練過程
    train_losses = []
    train_accs = []
    train_f1s = []
    val_losses = []
    val_accs = []
    val_f1s = []
    
    # 訓練
    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    
    # 測試
    print(f'\n========== Testing {model_name} ==========')
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(model, test_loader, criterion, device)
    print(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
    
    # 儲存模型
    torch.save(model.state_dict(), f'{model_name}_model.pth')
    print(f'Model saved as {model_name}_model.pth')
    
    return {
        'model_name': model_name,
        'train_accs': train_accs,
        'train_f1s': train_f1s,
        'val_accs': val_accs,
        'val_f1s': val_f1s,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_preds': test_preds,
        'test_labels': test_labels
    }
#%%
# ==================== 步驟 6: 訓練兩個模型 ====================
results = []
results.append(train_model('ResNet18', num_epochs=10))
results.append(train_model('ResNet50', num_epochs=10))
#%%
# ==================== 步驟 7: 繪製訓練曲線（比較兩個模型）====================
plt.figure(figsize=(15, 5))

# Training Accuracy
plt.subplot(1, 3, 1)
for result in results:
    plt.plot(result['train_accs'], label=result['model_name'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison')
plt.legend()
plt.grid(True)

# Training F1-Score
plt.subplot(1, 3, 2)
for result in results:
    plt.plot(result['train_f1s'], label=result['model_name'])
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('Training F1-Score Comparison')
plt.legend()
plt.grid(True)

# Test Results
plt.subplot(1, 3, 3)
models_names = [r['model_name'] for r in results]
test_accs = [r['test_acc'] for r in results]
test_f1s = [r['test_f1'] for r in results]
x = np.arange(len(models_names))
width = 0.35
plt.bar(x - width/2, test_accs, width, label='Accuracy')
plt.bar(x + width/2, test_f1s, width, label='F1-Score')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Test Results Comparison')
plt.xticks(x, models_names)
plt.legend()
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('training_comparison.png', dpi=150)
plt.show()
#%%
# ==================== 步驟 8: 繪製混淆矩陣 ====================
for result in results:
    cm = confusion_matrix(result['test_labels'], result['test_preds'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title(f'Confusion Matrix - {result["model_name"]}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{result["model_name"]}.png', dpi=150)
    plt.show()
#%%
# ==================== 步驟 9: 輸出最終結果 ====================
print('\n========== Final Results ==========')
for result in results:
    print(f'{result["model_name"]}:')
    print(f'  Test Accuracy: {result["test_acc"]:.4f}')
    print(f'  Test F1-Score: {result["test_f1"]:.4f}')
    print()

print('Training completed!')
