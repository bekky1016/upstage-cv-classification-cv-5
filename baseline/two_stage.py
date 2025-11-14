#!/usr/bin/env python
# coding: utf-8

"""
2-Stage Classification Pipeline
- Stage 1: 4개 그룹 분류 (Photo / ID / Document1 / Document2)
- Stage 2: 각 그룹 내부 세부 클래스 분류
- Backbone: efficientnet_b3
- K-Fold + AMP + Early Stopping + TTA + K-Fold Ensemble
"""

import os
import random
import datetime

import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from timm.loss import LabelSmoothingCrossEntropy

# =========================
# 1. 환경 및 설정
# =========================

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 경로 (필요시 여기만 수정)
TRAIN_CSV = "../data/train_mod_balanced.csv"   # ID,target
TRAIN_IMG_DIR = "../data/train_mod_balanced/"  # 이미지 폴더
TEST_CSV = "../data/sample_submission.csv"     # ID,target(dummy)
TEST_IMG_DIR = "../data/test/"

# 공통 하이퍼파라미터
MODEL_NAME = "efficientnet_b3"
IMG_SIZE = 512
N_FOLDS = 5

LR = 1e-4
EPOCHS = 60
BATCH_SIZE = 32
NUM_WORKERS = 4
PATIENCE = 10  # Early Stopping patience

USE_WANDB = True
WANDB_PROJECT = "document-type-classification-2stage"


# =========================
# 2. 클래스 → 그룹 매핑 정의
# =========================

# 원본 클래스 ID (0~16)
# 0	account_number
# 1	application_for_payment_of_pregnancy_medical_expenses
# 2	car_dashboard
# 3	confirmation_of_admission_and_discharge
# 4	diagnosis
# 5	driver_license
# 6	medical_bill_receipts
# 7	medical_outpatient_certificate
# 8	national_id_card
# 9	passport
# 10	payment_confirmation
# 11	pharmaceutical_receipt
# 12	prescription
# 13	resume
# 14	statement_of_opinion
# 15	vehicle_registration_certificate
# 16	vehicle_registration_plate

# ✅ 그룹 매핑 (시각적 패턴 기준)
# G0 (0): Photo-type       -> 2, 16
# G1 (1): ID-type          -> 5, 8, 9
# G2 (2): Document-type    -> 나머지 문서 (0,1,3,4,6,7,10,11,12,13,14,15)

# group_map = {
#     2: 0, 16: 0,                                 # Photo-type
#     5: 1, 8: 1, 9: 1,                            # ID-type
#     0: 2, 1: 2, 3: 2, 4: 2, 6: 2, 7: 2,
#     10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2     # Document-type
# }
group_map = {
    2: 0, 16: 0,                                 # G0: Photo-type
    5: 1, 8: 1, 9: 1,                            # G1: ID-type
    3: 2, 7: 2, 14: 2,                           # G2: Document-type-1
    0: 3, 1: 3, 4: 3, 6: 3, 10: 3, 11: 3, 12: 3, 13: 3, 15: 3  # G3: Document-type-2
}

# group_to_classes = {
#     0: [2, 16],                                  # Photo-type
#     1: [5, 8, 9],                                # ID-type
#     2: [0, 1, 3, 4, 6, 7, 10, 11, 12, 13, 14, 15]  # Document-type
# }
group_to_classes = {
    0: [2, 16],                     # Photo-type
    1: [5, 8, 9],                   # ID-type
    2: [3, 7, 14],                  # Document-type-1
    3: [0, 1, 4, 6, 10, 11, 12, 13, 15]  # Document-type-2
}


# =========================
# 3. Dataset 정의
# =========================

class ImageDataset(Dataset):
    """
    df: 반드시 [ID, label] 순서의 두 컬럼만 가진 DataFrame을 넣는다고 가정
        (label = target 또는 group)
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True).values  # [[ID, label], ...]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name, label = self.df[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, int(label)


# =========================
# 4. Transform 정의
# =========================

def get_train_transform(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=180, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomResizedCrop(height=img_size, width=img_size,
                            scale=(0.8, 1.0), p=0.4),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_valid_transform(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# =========================
# 5. Train / Valid Loop
# =========================

def train_one_epoch(loader, model, optimizer, loss_fn, device, epoch=None,
                    scaler=None, use_amp=True):
    model.train()
    train_loss = 0.0
    preds_list, targets_list = [], []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}" if epoch is not None else "Train")
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast():
                preds = model(images)
                loss = loss_fn(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    # wandb 로그
    log_dict = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }
    if epoch is not None:
        log_dict["epoch"] = epoch + 1
    wandb.log(log_dict)

    return train_loss, train_acc, train_f1


def valid_one_epoch(loader, model, loss_fn, device, epoch=None, num_classes=None, class_names=None):
    model.eval()
    val_loss = 0.0
    preds_list, targets_list = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Valid Epoch {epoch+1}" if epoch is not None else "Valid")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            loss = loss_fn(preds, targets)

            val_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(targets.detach().cpu().numpy())

    val_loss /= len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')

    # ✅ Confusion Matrix 계산
    if num_classes is None:
        num_classes = len(set(targets_list))
    cm = confusion_matrix(targets_list, preds_list, labels=range(num_classes))

    # ✅ Matplotlib + Seaborn 시각화
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names if class_names else range(num_classes),
                yticklabels=class_names if class_names else range(num_classes))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (Epoch {epoch+1})')

    # ✅ wandb에 이미지로 업로드
    wandb.log({
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "epoch": epoch + 1 if epoch is not None else 0,
        "confusion_matrix": wandb.Image(fig)
    })

    plt.close(fig)

    return val_loss, val_acc, val_f1


# =========================
# 6. Stage 1: Group 분류 학습
# =========================

def train_stage1_group_model(train_df):
    """
    train_df: 원본 train DataFrame (ID, target)
    -> group_df를 만들어 3개 그룹 분류 모델 학습 (K-Fold)
    -> best 모델 경로 리스트 반환
    """
    print("\n===== Stage 1: Group Classification (3 classes) =====")

    # group 라벨 생성
    group_df = train_df.copy()
    group_df["group"] = group_df["target"].map(group_map)
    assert group_df["group"].isna().sum() == 0, "group_map에 없는 클래스가 있습니다."

    # ID, group만 사용
    group_df = group_df[["ID", "group"]]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    group_model_paths = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(group_df, group_df["group"])):
        print(f"\n--- [Stage1] Fold {fold+1}/{N_FOLDS} ---")

        trn_df = group_df.iloc[tr_idx].reset_index(drop=True)
        val_df = group_df.iloc[val_idx].reset_index(drop=True)

        train_transform = get_train_transform(IMG_SIZE)
        valid_transform = get_valid_transform(IMG_SIZE)

        trn_dataset = ImageDataset(trn_df, TRAIN_IMG_DIR, transform=train_transform)
        val_dataset = ImageDataset(val_df, TRAIN_IMG_DIR, transform=valid_transform)

        trn_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        num_classes = 4

        model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=LR)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        loss_fn = nn.CrossEntropyLoss()
        # loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1) # 라벨스무딩 쓰려면 loss_fn = nn.CrossEntropyLoss() 주석
        scaler = GradScaler()

        run_name = f"{MODEL_NAME}_STAGE1_group_fold{fold+1}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        if USE_WANDB:
            wandb.init(project=WANDB_PROJECT, name=run_name)

        best_val_f1 = 0.0
        counter = 0
        best_path = f"{MODEL_NAME}_stage1_group_fold{fold+1}_best.pt"

        for epoch in range(EPOCHS):
            torch.cuda.empty_cache()
            print(f"\n[Stage1][Fold {fold+1}] Epoch {epoch+1}/{EPOCHS}")

            train_loss, train_acc, train_f1 = train_one_epoch(
                trn_loader, model, optimizer, loss_fn, device, epoch=epoch,
                scaler=scaler, use_amp=True
            )
            val_loss, val_acc, val_f1 = valid_one_epoch(
                val_loader, model, loss_fn, device, epoch=epoch,
                num_classes=num_classes,  # ✅ 추가
                class_names=[str(i) for i in range(num_classes)]  # ✅ 추가
            )

            scheduler.step()

            print(f"[Stage1][Fold {fold+1}] "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

            # Early Stopping 체크
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                counter = 0
                torch.save(model.state_dict(), best_path)
                print(f"🌟 [Stage1][Fold {fold+1}] Best 업데이트! Val F1={val_f1:.4f}")
            else:
                counter += 1
                print(f"⚠️ [Stage1][Fold {fold+1}] 개선 없음 ({counter}/{PATIENCE})")
                if counter >= PATIENCE:
                    print(f"⏹️ [Stage1][Fold {fold+1}] Early Stopping 발동")
                    break

        group_model_paths.append(best_path)
        if USE_WANDB:
            wandb.finish()

    return group_model_paths


# =========================
# 7. Stage 2: 그룹별 세부 클래스 분류 학습
# =========================

def train_stage2_per_group(train_df):
    """
    train_df: 원본 train DataFrame (ID, target)
    -> 각 group별로 K-Fold 세부 분류 모델 학습
    -> dict[group_id] = [모델경로들] 반환
    """
    print("\n===== Stage 2: Per-Group Fine Classification =====")

    group_to_model_paths = {}

    for g, cls_list in group_to_classes.items():
        print(f"\n### [Stage2] Group {g} (classes: {cls_list}) ###")

        df_g = train_df[train_df["target"].isin(cls_list)].reset_index(drop=True)
        print(f"[Group {g}] 데이터 수: {len(df_g)}")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        paths_g = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(df_g, df_g["target"])):
            print(f"\n--- [Stage2] Group {g} Fold {fold+1}/{N_FOLDS} ---")

            trn_df = df_g.iloc[tr_idx][["ID", "target"]].reset_index(drop=True)
            val_df = df_g.iloc[val_idx][["ID", "target"]].reset_index(drop=True)

            train_transform = get_train_transform(IMG_SIZE)
            valid_transform = get_valid_transform(IMG_SIZE)

            trn_dataset = ImageDataset(trn_df, TRAIN_IMG_DIR, transform=train_transform)
            val_dataset = ImageDataset(val_df, TRAIN_IMG_DIR, transform=valid_transform)

            trn_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

            num_classes = len(cls_list)

            model = timm.create_model(MODEL_NAME, pretrained=True,
                                      num_classes=num_classes).to(device)
            optimizer = Adam(model.parameters(), lr=LR)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
            loss_fn = nn.CrossEntropyLoss()
            # loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1) # 라벨스무딩 쓰려면 loss_fn = nn.CrossEntropyLoss() 주석

            scaler = GradScaler()

            run_name = f"{MODEL_NAME}_STAGE2_g{g}_fold{fold+1}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
            if USE_WANDB:
                wandb.init(project=WANDB_PROJECT, name=run_name)

            best_val_f1 = 0.0
            counter = 0
            best_path = f"{MODEL_NAME}_stage2_g{g}_fold{fold+1}_best.pt"

            # target을 group 내 index(0~k-1)로 바꾸기 위한 매핑
            # 예: cls_list = [5,8,9] -> global 5->0, 8->1, 9->2
            class_to_local = {c: i for i, c in enumerate(cls_list)}

            # 로더에서 라벨 변환을 위해 간단한 래퍼 쓸 수도 있지만,
            # 여기서는 loss 계산 부분에서 직접 변환하는 대신
            # train_df / val_df 쪽 target을 local로 바꿔서 dataset에 넣는 것이 깔끔함
            trn_df_local = trn_df.copy()
            val_df_local = val_df.copy()
            trn_df_local["target"] = trn_df_local["target"].map(class_to_local)
            val_df_local["target"] = val_df_local["target"].map(class_to_local)

            trn_dataset = ImageDataset(trn_df_local, TRAIN_IMG_DIR, transform=train_transform)
            val_dataset = ImageDataset(val_df_local, TRAIN_IMG_DIR, transform=valid_transform)

            trn_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

            for epoch in range(EPOCHS):
                torch.cuda.empty_cache()
                print(f"\n[Stage2][Group {g}] [Fold {fold+1}] Epoch {epoch+1}/{EPOCHS}")

                train_loss, train_acc, train_f1 = train_one_epoch(
                    trn_loader, model, optimizer, loss_fn, device, epoch=epoch,
                    scaler=scaler, use_amp=True
                )
                val_loss, val_acc, val_f1 = valid_one_epoch(
                    val_loader, model, loss_fn, device, epoch=epoch,
                    num_classes=num_classes,  # ✅ 추가
                    class_names=[str(i) for i in range(num_classes)]  # ✅ 추가
                )
                scheduler.step()

                print(f"[Stage2][Group {g}] [Fold {fold+1}] "
                      f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    counter = 0
                    torch.save(model.state_dict(), best_path)
                    print(f"🌟 [Stage2][Group {g}][Fold {fold+1}] Best 업데이트! Val F1={val_f1:.4f}")
                else:
                    counter += 1
                    print(f"⚠️ [Stage2][Group {g}][Fold {fold+1}] 개선 없음 ({counter}/{PATIENCE})")
                    if counter >= PATIENCE:
                        print(f"⏹️ [Stage2][Group {g}][Fold {fold+1}] Early Stopping 발동")
                        break

            paths_g.append(best_path)
            if USE_WANDB:
                wandb.finish()

        group_to_model_paths[g] = paths_g

    return group_to_model_paths


# =========================
# 8. TTA 함수
# =========================

tta_transforms = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[3]),  # 좌우
    lambda x: torch.flip(x, dims=[2]),  # 상하
    lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # 90도
    lambda x: torch.rot90(x, k=3, dims=[2, 3])   # -90도
]

def tta_predict_probs(model, loader, num_classes):
    """
    TTA + softmax 확률 평균
    return: (N, num_classes) numpy array
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="TTA Inference"):
            images = images.to(device)
            tta_probs = []

            for tta in tta_transforms:
                imgs_tta = tta(images)
                preds = model(imgs_tta)
                probs = preds.softmax(dim=1).cpu().numpy()
                tta_probs.append(probs)

            avg_probs = np.mean(tta_probs, axis=0)  # (B, num_classes)
            all_probs.append(avg_probs)

    all_probs = np.concatenate(all_probs, axis=0)
    return all_probs  # (N, num_classes)


# =========================
# 9. 2-Stage Inference (Stage1 + Stage2 + Gating)
# =========================

def inference_two_stage(group_model_paths, group_to_model_paths):
    print("\n===== 2-Stage Inference 시작 =====")

    # Test Dataset / Loader
    test_df = pd.read_csv(TEST_CSV)   # ID, target(dummy)
    test_ids = test_df["ID"].values

    test_transform = get_valid_transform(IMG_SIZE)
    # dummy label 0
    test_pairs = pd.DataFrame({"ID": test_ids, "dummy": [0] * len(test_ids)})
    test_dataset = ImageDataset(test_pairs[["ID", "dummy"]], TEST_IMG_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)

    N = len(test_dataset)
    num_groups = 4

    # ----- Stage1: 그룹 확률 -----
    print("\n[Inference] Stage1 Group Probabilities")
    stage1_probs = np.zeros((N, num_groups), dtype=np.float32)

    for path in group_model_paths:
        print(f" ▶ Loading Stage1 model: {path}")
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_groups).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        probs = tta_predict_probs(model, test_loader, num_classes=num_groups)
        stage1_probs += probs

    stage1_probs /= len(group_model_paths)  # K-Fold 평균


    # ----- Stage2: 그룹별 세부 클래스 확률 -----
    print("\n[Inference] Stage2 Group-wise Class Probabilities")

    # 그룹별: (N, len(cls_list))
    stage2_probs_per_group = {}
    for g, cls_list in group_to_classes.items():
        print(f"\n ▶ Group {g} (classes: {cls_list})")
        num_classes_g = len(cls_list)
        probs_g = np.zeros((N, num_classes_g), dtype=np.float32)

        for path in group_to_model_paths[g]:
            print(f"   - Loading Stage2 model: {path}")
            model = timm.create_model(MODEL_NAME, pretrained=False,
                                      num_classes=num_classes_g).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            p = tta_predict_probs(model, test_loader, num_classes=num_classes_g)
            probs_g += p

        probs_g /= len(group_to_model_paths[g])  # group 내 K-Fold 평균
        stage2_probs_per_group[g] = probs_g


    # ----- 2-Stage Gating: P(class) = P(group) * P(class|group) -----
    print("\n[Inference] Combining Stage1 & Stage2 (Gating)")

    num_global_classes = 17
    final_probs = np.zeros((N, num_global_classes), dtype=np.float32)

    for g, cls_list in group_to_classes.items():
        # group g에 대한 P(group = g | x)
        pg = stage1_probs[:, g].reshape(-1, 1)  # (N, 1)
        # group g 내부 클래스 확률: P(class | group=g, x)
        pcg = stage2_probs_per_group[g]         # (N, len(cls_list))

        # global class index로 매핑
        for local_idx, global_cls in enumerate(cls_list):
            final_probs[:, global_cls] += pg[:, 0] * pcg[:, local_idx]

    # 정규화 (이론상 이미 1이지만 수치 보정)
    final_probs /= final_probs.sum(axis=1, keepdims=True)
    final_preds = np.argmax(final_probs, axis=1)

    # ----- CSV 저장 -----
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    save_path = f"pred_2stage_{MODEL_NAME}_{timestamp}.csv"

    out_df = pd.read_csv(TEST_CSV)
    out_df["target"] = final_preds
    out_df.to_csv(save_path, index=False)
    print(f"\n✅ Saved 2-Stage prediction: {save_path}")


# =========================
# 10. main()
# =========================

def main():
    # 1) Train 데이터 로드
    train_df = pd.read_csv(TRAIN_CSV)  # ID, target
    print("Train shape:", train_df.shape)
    print(train_df["target"].value_counts().sort_index())

    # 2) Stage1: Group 분류 모델 학습 (3 클래스)
    group_model_paths = train_stage1_group_model(train_df)

    # 3) Stage2: 그룹별 세부 클래스 모델 학습
    group_to_model_paths = train_stage2_per_group(train_df)

    # 4) 2-Stage Inference (TTA + Ensemble + Gating)
    inference_two_stage(group_model_paths, group_to_model_paths)


if __name__ == "__main__":
    main()
