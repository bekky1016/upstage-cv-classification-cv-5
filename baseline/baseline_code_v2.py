#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Library & Define Functions
# * 학습 및 추론에 필요한 라이브러리를 로드합니다.
# * 학습 및 추론에 필요한 함수와 클래스를 정의합니다.

# In[ ]:


import os
import time
import timm
import torch
import wandb
import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from timm.loss import LabelSmoothingCrossEntropy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# In[ ]:


# 시드를 고정합니다.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True


# In[ ]:


# 데이터셋 클래스를 정의합니다.
class ImageDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df.values if isinstance(df, pd.DataFrame) else pd.read_csv(df).values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target


# In[ ]:

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler() # fold 루프 전에서 초기화해야 재사용 가능

# one epoch 학습을 위한 함수입니다.
def train_one_epoch(loader, model, optimizer, loss_fn, device, epoch=None):
    model.train()
    train_loss = 0
    preds_list, targets_list = [], []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}" if epoch is not None else "Train")
    for image, targets in pbar:
        image, targets = image.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        # ✅ 자동 혼합 정밀도 (AMP)
        with autocast(): # forward 연산을 반정밀도로 계산 (메모리 절약)
            preds = model(image)
            loss = loss_fn(preds, targets)

        # ✅ Scaler로 역전파
        scaler.scale(loss).backward() # 손실값을 안전하게 스케일링
        scaler.step(optimizer) # 옵티마이저 업데이트 시 overflow 방지
        scaler.update() # 옵티마이저 업데이트 시 overflow 방지
        
        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).cpu().numpy())
        targets_list.extend(targets.cpu().numpy())

    # ---- epoch별 평균 계산 ----
    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    # ---- wandb 로그 기록 ----
    wandb.log({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "lr": optimizer.param_groups[0]["lr"],        # ✅ 학습률 로그 추가
        "epoch": epoch + 1 if epoch is not None else 0
    })

    return {"train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1}


# Validation용 함수 추가
def valid_one_epoch(loader, model, loss_fn, device, epoch=None, fold=None):
    model.eval()
    val_loss = 0
    preds_list, targets_list = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Valid Epoch {epoch+1}" if epoch is not None else "Valid")
        for image, targets in pbar:
            image, targets = image.to(device), targets.to(device)
            preds = model(image)
            loss = loss_fn(preds, targets)
            val_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    val_loss /= len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')

    # 🟩 1️⃣ Confusion Matrix 계산
    cm = confusion_matrix(targets_list, preds_list)

    # 🟩 2️⃣ 클래스별 정확도 계산
    class_acc = cm.diagonal() / cm.sum(axis=1)
    class_acc_dict = {f"class_{i}_acc": float(acc) for i, acc in enumerate(class_acc)}

    # 🟩 3️⃣ 시각화 및 wandb 업로드
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - Fold {fold+1 if fold is not None else '?'} (Epoch {epoch+1})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # ---- wandb 로그 기록 ----
    wandb.log({
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        **class_acc_dict,  # 🟩 클래스별 acc도 같이 기록
        "epoch": epoch + 1 if epoch is not None else 0,
        "confusion_matrix": wandb.Image(plt)  # 🟩 시각화 이미지 로그
    })
    plt.close()

    return {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "preds": preds_list, "targets": targets_list}


# ## 2. Hyper-parameters
# * 학습 및 추론에 필요한 하이퍼파라미터들을 정의합니다.

# In[ ]:


# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data config
data_path = 'datasets_fin/'

# model config
# model_name = 'resnet34' # 'resnet50' 'efficientnet-b0', ...
# model_name = 'efficientnet_b3'
# model_name = 'convnext_tiny'
# model_name = 'vit_base_patch16_384'
# model_name = 'swin_base_patch4_window12_384'
# model_name = 'resnext50_32x4d'
model_name = 'resnext101_32x8d'

# training config
img_size = 384 # 224, 384, 640
LR = 1e-4 #  3e-4 < 1e-3
EPOCHS = 100
BATCH_SIZE = 32 # 32
num_workers = 4 # 4


# ## 3. Load Data
# * 학습, 테스트 데이터셋과 로더를 정의합니다.

# In[ ]:


# augmentation을 위한 transform 코드
trn_transform = A.Compose([
    # 이미지 크기 조정
    A.Resize(height=img_size, width=img_size),
    
    # --- 실제 Test domain 대응 증강 ---
    A.Rotate(limit=180, p=0.7),                     # 회전
    A.HorizontalFlip(p=0.5),                        # 좌우 반전
    A.VerticalFlip(p=0.3),                          # 상하 반전
    A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), p=0.4),  # 크롭
    A.MotionBlur(blur_limit=5, p=0.3),              # 블러
    A.GaussNoise(var_limit=(10, 50), p=0.3),        # 노이즈
    A.RandomBrightnessContrast(p=0.3),              # 밝기/대비
    A.HueSaturationValue(p=0.2),                    # 색조 변형 (인쇄/조명 차이 대응)

    # 기본적인 뒤집기 + 살짝 회전만
    # A.HorizontalFlip(p=0.5),
    # A.Rotate(limit=15, p=0.3),

    # 너무 심한 크롭/노이즈/블러는 일단 제거
    # A.RandomBrightnessContrast(p=0.2),
    
    # images normalization
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # numpy 이미지나 PIL 이미지를 PyTorch 텐서로 변환
    ToTensorV2(),
])

# test image 변환을 위한 transform 코드
tst_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# In[ ]:


# print("총 이미지 수:", len(os.listdir("../data/train_balanced")))
print("총 이미지 수:", len(os.listdir("../data/train_mod_balanced")))
# df = pd.read_csv("../data/train_balanced.csv")
df = pd.read_csv("../data/train_mod_balanced.csv")
print(df["target"].value_counts().sort_index())


# In[ ]:


# --- ✅ K-Fold split으로 변경 ---
# train_df = pd.read_csv("../data/train_balanced.csv")
train_df = pd.read_csv("../data/train_mod_balanced.csv")

folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

# ✅ progressive resizing 헬퍼 추가
def adjust_img_size(epoch):
    # 이미지 사이즈 고정
    return 384
    # return 640
    # 폴드 기반
    # if fold < 2:
    #     return 384
    # elif fold < 4:
    #     return 512
    # else:
    #     return 640
    # 에포크 기반
    # if epoch < 30:
    #     return 384
    # elif epoch < 45:
    #     return 512
    # else:
    #     return 640

def update_transforms(new_size):
    global trn_transform, tst_transform
    trn_transform = A.Compose([
        A.Resize(height=new_size, width=new_size),
        A.Rotate(limit=90, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(height=new_size, width=new_size, scale=(0.9, 1.0), p=0.3),
        A.GaussNoise(var_limit=(10, 40), p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    tst_transform = A.Compose([
        A.Resize(height=new_size, width=new_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ✅ fold별 학습 루프
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
    new_size = adjust_img_size(fold)
    update_transforms(new_size)
    # print(f"\n===== Fold {fold+1}/{folds} =====")
    print(f"\n===== Fold {fold+1}/{folds} | 이미지 크기: {new_size}px =====")

    trn_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    # trn_dataset = ImageDataset(trn_df, "../data/train_balanced/", transform=trn_transform)
    # val_dataset = ImageDataset(val_df, "../data/train_balanced/", transform=tst_transform)
    trn_dataset = ImageDataset(trn_df, "../data/train_mod_balanced/", transform=trn_transform)
    val_dataset = ImageDataset(val_df, "../data/train_mod_balanced/", transform=tst_transform)

    trn_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = timm.create_model(model_name, pretrained=True, num_classes=17).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    loss_fn = nn.CrossEntropyLoss()
    # 위 loss_fn = nn.CrossEntropyLoss() 주석처리 하고 아래 주석 풀면 라벨스무딩 사용 가능
    # loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)

    run_name = f"{model_name}_fold{fold+1}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(project="document-type-classification", name=run_name)

    # ✅ Early Stopping 설정
    best_val_f1 = 0
    patience = 15     # 개선 안 되는 에폭이 5번 연속이면 stop
    counter = 0 
    
    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()  # ✅ 캐시 해제
        print(f"\n[Fold {fold+1}] [Epoch {epoch+1}]")  # 크기 고정
        all_preds, all_targets = [], []

        train_metrics = train_one_epoch(trn_loader, model, optimizer, loss_fn, device=device, epoch=epoch)
        val_metrics = valid_one_epoch(val_loader, model, loss_fn, device=device, epoch=epoch, fold=fold)
        all_preds.extend(val_metrics["preds"])
        all_targets.extend(val_metrics["targets"])
        
        scheduler.step()

        val_f1 = val_metrics["val_f1"]

        # ✅ best 모델 저장 로직
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"{model_name}_fold{fold+1}_best.pt")
            print(f"🌟 개선됨! 모델 저장 (Val F1={val_f1:.4f})")
            counter = 0  # patience 초기화
        else:
            counter += 1
            print(f"⚠️ 개선 없음 ({counter}/{patience})")

        if counter >= patience:
            print(f"⏹️ Early Stopping 발동 (최대 {patience}회 미개선)")
            break
        
        print(
            f"[Fold {fold+1}] [Epoch {epoch+1}/{EPOCHS}] "
            f"Train F1: {train_metrics['train_f1']:.4f}, "
            f"Val F1: {val_metrics['val_f1']:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.8f}"
        )

    model_path = f"{model_name}_fold{fold+1}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ 모델 저장 완료: {model_path}")
    wandb.finish()

# 각 fold의 validation 결과를 저장하고 병합해서 confusion matrix 보여주기
cm_total = confusion_matrix(all_targets, all_preds)
sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - All Folds")
plt.show()

# ## 4. Train Model
# * 모델을 로드하고, 학습을 진행합니다.

# # 5. Inference & Save File
# * 테스트 이미지에 대한 추론을 진행하고, 결과 파일을 저장합니다.

# In[ ]:


# --- ✅ 모든 Fold 학습 완료 후 Inference 단계 ---
print("\n===== TTA Inference 시작 =====")

# ✅ test dataset / loader 정의
tst_dataset = ImageDataset(
    "../data/sample_submission.csv",
    "../data/test/",
    transform=tst_transform
)
tst_loader = DataLoader(tst_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ✅ fold별 모델 경로 지정 (이미 학습 완료된 모델들)
model_paths = [f"{model_name}_fold{i+1}_best.pt" for i in range(folds)]

tta_transforms = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[3]),
    lambda x: torch.flip(x, dims=[2]),
    lambda x: torch.rot90(x, k=1, dims=[2,3]),
    lambda x: torch.rot90(x, k=3, dims=[2,3])
]

preds_all = []
for path in model_paths:
    print(f"\n▶ Loading {path}")
    # fold마다 새 모델 객체를 만들어주고 weight를 로드
    # 이미 학습한 .pt 파일에 가중치가 있으니까, 다시 ImageNet weight 로 불러올 필요 없음
    model = timm.create_model(model_name, pretrained=False, num_classes=17).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    preds_fold = []
    for images, _ in tqdm(tst_loader):
        images = images.to(device)
        tta_preds = []

        with torch.no_grad():
            for tta in tta_transforms:
                imgs_tta = tta(images)
                preds = model(imgs_tta)
                tta_preds.append(preds.softmax(dim=1).cpu().numpy())

        avg_preds = np.mean(tta_preds, axis=0)
        preds_fold.append(avg_preds)

    preds_fold = np.concatenate(preds_fold)
    preds_all.append(preds_fold)

# ✅ K-Fold 평균 앙상블
avg_preds = np.mean(preds_all, axis=0)
final_preds = np.argmax(avg_preds, axis=1)

# ✅ 현재 시간 기반 파일명 생성
timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
save_path = f"pred_{timestamp}.csv"

# ✅ 결과 저장
tst_df = pd.read_csv("../data/sample_submission.csv")
tst_df["target"] = final_preds
tst_df.to_csv(save_path, index=False)

print(f"✅ Saved submission: {save_path}")


# %%
