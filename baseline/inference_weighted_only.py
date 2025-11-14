#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import timm
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image

# ============================================================
# 1. Dataset 정의
# ============================================================
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

# ============================================================
# 2. 기본 설정
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'efficientnet_b3'
folds = 5
BATCH_SIZE = 32
img_size = 640

# ============================================================
# 3. Transform 정의
# ============================================================
tst_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ============================================================
# 4. Test 데이터 로드
# ============================================================
tst_dataset = ImageDataset(
    "../data/sample_submission.csv",
    "../data/test/",
    transform=tst_transform
)
tst_loader = DataLoader(tst_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# 5. Fold별 Validation F1 입력 (가중치 계산용)
# ============================================================
# ⚠️ 수동 입력 (val_f1이 높은 순서대로 입력)
# 예: fold2가 월등히 좋았다면 아래처럼 조정
fold_f1s = [0.95592, 0.96047, 0.95441, 0.95544, 0.95837]

weights = np.array(fold_f1s) / np.sum(fold_f1s)
print(f"📊 Fold별 가중치 (합=1): {np.round(weights, 4)}")

# ============================================================
# 6. TTA 정의
# ============================================================
tta_transforms = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[3]),
    lambda x: torch.flip(x, dims=[2]),
    lambda x: torch.rot90(x, k=1, dims=[2,3]),
    lambda x: torch.rot90(x, k=3, dims=[2,3])
]

# ============================================================
# 7. Inference (Weighted Ensemble)
# ============================================================
model_paths = [f"{model_name}_fold{i+1}_best.pt" for i in range(folds)]
preds_all = []

for fold_idx, path in enumerate(model_paths):
    if not os.path.exists(path):
        print(f"⚠️ 경고: {path} 파일이 존재하지 않음. 이 fold는 건너뜁니다.")
        continue

    print(f"\n▶ Loading {path} (Weight={weights[fold_idx]:.4f})")
    model = timm.create_model(model_name, pretrained=False, num_classes=17).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    preds_fold = []
    for images, _ in tqdm(tst_loader, desc=f"Inference Fold {fold_idx+1}"):
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

# ============================================================
# 8. Weighted 평균 앙상블
# ============================================================
avg_preds = np.tensordot(preds_all, weights, axes=((0), (0)))
final_preds = np.argmax(avg_preds, axis=1)

# ============================================================
# 9. 결과 저장
# ============================================================
timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
save_path = f"pred_weighted_{timestamp}.csv"

tst_df = pd.read_csv("../data/sample_submission.csv")
tst_df["target"] = final_preds
tst_df.to_csv(save_path, index=False)

print(f"\n✅ Weighted Ensemble 결과 저장 완료 → {save_path}")
print(f"📈 Fold별 가중치: {np.round(weights, 4)}")
