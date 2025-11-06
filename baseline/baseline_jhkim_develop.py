"""ResNeXt50 기반 문서 타입 분류 학습/추론 스크립트.

원본 노트북(baseline_jhkim.ipynb)의 ResNeXt50 + Stratified K-Fold
파이프라인을 그대로 재현하면서, 초보자도 이해하기 쉽도록
함수 단위로 모듈화하고 주석을 자세히 달았습니다.

이 스크립트는 아래 순서로 동작합니다.
1. 재현 가능한 실험을 위해 모든 난수를 고정합니다.
2. 원본 학습 이미지를 불러와 오프라인 증강 이미지를 생성합니다.
3. Albumentations 기반 변환(학습/추론용)을 정의합니다.
4. 증강된 학습 데이터셋과 테스트 데이터셋을 구성합니다.
5. Stratified K-Fold(기본 5-Fold)로 ResNeXt50 모델을 학습합니다.
6. 각 Fold 모델을 이용해 테스트 세트 예측을 Softmax 평균 앙상블합니다.
7. 최종 submission.csv 파일을 저장합니다.

ConvNeXt Tiny / EfficientNet-B1 실험 코드는 제외했습니다.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


# -------------------------------------------------------------
# 1. 실험 재현성을 위한 시드 고정 함수
# -------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """모든 주요 라이브러리의 난수를 고정합니다."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# -------------------------------------------------------------
# 2. Dataset 정의
# -------------------------------------------------------------
class DocumentDataset(Dataset):
    """CSV 파일과 이미지 폴더를 읽어 PyTorch Dataset으로 제공합니다."""

    def __init__(self, csv_path: str, image_root: str, transform: A.Compose | None = None) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

        # CSV 컬럼명이 데이터셋마다 다르므로 공통 포맷으로 정리합니다.
        if "ID" in self.df.columns:
            self.image_names = self.df["ID"].tolist()
        elif "filename" in self.df.columns:
            self.image_names = self.df["filename"].tolist()
        else:
            raise ValueError("CSV에 이미지 파일명을 나타내는 'ID' 또는 'filename' 컬럼이 필요합니다.")

        if "target" in self.df.columns:
            self.labels = self.df["target"].tolist()
        elif "label" in self.df.columns:
            self.labels = self.df["label"].tolist()
        else:
            # 테스트 셋처럼 라벨이 없는 경우 0으로 채워 둡니다.
            self.labels = [0] * len(self.df)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_name = self.image_names[index]
        label = int(self.labels[index])

        image_path = os.path.join(self.image_root, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

        # OpenCV는 BGR 순서로 읽기 때문에 RGB로 변환합니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label


# -------------------------------------------------------------
# 3. 오프라인 증강 이미지 생성
# -------------------------------------------------------------
def create_offline_augmented_dataset(
    input_csv: str,
    input_dir: str,
    output_dir: str,
    output_csv: str,
    num_augments: int = 4,
) -> None:
    """원본 이미지를 복사하고 추가 증강본을 생성해 CSV로 기록합니다."""

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    augmented_records: List[dict] = []

    augment = A.Compose(
        [
            A.Rotate(limit=180, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.5),
        ]
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="오프라인 증강 생성"):
        image_name = row["ID"]
        label = int(row["target"])
        image_path = os.path.join(input_dir, image_name)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

        # 1) 원본 이미지를 증강 폴더로 복사(덮어쓰기 허용)
        original_save_path = os.path.join(output_dir, image_name)
        cv2.imwrite(original_save_path, image)
        augmented_records.append({"filename": image_name, "label": label})

        # 2) 추가 증강본 생성
        for aug_idx in range(num_augments):
            augmented = augment(image=image)["image"]
            new_name = f"{os.path.splitext(image_name)[0]}_aug{aug_idx + 1}.jpg"
            cv2.imwrite(os.path.join(output_dir, new_name), augmented)
            augmented_records.append({"filename": new_name, "label": label})

    aug_df = pd.DataFrame(augmented_records)
    aug_df.to_csv(output_csv, index=False)

    print(f"✅ 오프라인 증강 CSV 저장 완료: {output_csv}")


# -------------------------------------------------------------
# 4. Albumentations 변환 구성
# -------------------------------------------------------------
def build_transforms(image_size: int) -> Tuple[A.Compose, A.Compose]:
    """학습/추론에 사용할 Albumentations 변환을 반환합니다."""

    train_transform = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=180, p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                ],
                p=0.5,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return train_transform, test_transform


# -------------------------------------------------------------
# 5. 학습/검증 루프 유틸리티
# -------------------------------------------------------------
def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    """한 Epoch 동안 학습을 수행하고 주요 지표를 반환합니다."""

    model.train()
    epoch_loss = 0.0
    preds_list: List[int] = []
    targets_list: List[int] = []

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds_list.extend(logits.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

    epoch_loss /= len(loader)
    epoch_acc = accuracy_score(targets_list, preds_list)
    epoch_f1 = f1_score(targets_list, preds_list, average="macro")

    return {
        "train_loss": epoch_loss,
        "train_acc": epoch_acc,
        "train_f1": epoch_f1,
    }


def evaluate(val_loader: DataLoader, model: nn.Module, device: torch.device) -> float:
    """검증 데이터셋에 대한 Macro F1 스코어를 계산합니다."""

    model.eval()
    preds_list: List[int] = []
    targets_list: List[int] = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluate", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            preds_list.extend(logits.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(targets.detach().cpu().numpy())

    return f1_score(targets_list, preds_list, average="macro")


# -------------------------------------------------------------
# 6. K-Fold 학습 루틴
# -------------------------------------------------------------
@dataclass
class TrainingConfig:
    model_name: str = "resnext50_32x4d"
    image_size: int = 256
    batch_size: int = 32
    epochs: int = 30
    patience: int = 10
    lr: float = 5e-4
    num_workers: int = 4
    num_folds: int = 5


def train_kfold(
    dataset: DocumentDataset,
    config: TrainingConfig,
    device: torch.device,
) -> List[str]:
    """Stratified K-Fold 학습을 수행하고 Fold별 가중치 경로를 반환합니다."""

    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)

    fold_weight_paths: List[str] = []
    loss_fn = nn.CrossEntropyLoss()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), labels), start=1):
        print(f"\n========== Fold {fold}/{config.num_folds} ==========")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        model = timm.create_model(config.model_name, pretrained=True, num_classes=17).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

        best_val_f1 = 0.0
        patience_counter = 0
        best_path = f"best_model_fold{fold}.pth"

        for epoch in range(1, config.epochs + 1):
            train_metrics = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
            val_f1 = evaluate(val_loader, model, device)

            log_msg = (
                f"Epoch {epoch:03d} | "
                f"Loss {train_metrics['train_loss']:.4f} | "
                f"Acc {train_metrics['train_acc']:.4f} | "
                f"Train F1 {train_metrics['train_f1']:.4f} | "
                f"Val F1 {val_f1:.4f}"
            )
            print(log_msg)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
                print(f"→ 모델 개선! 현재 최고 Val F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(
                        f"Val F1가 {config.patience} Epoch 동안 향상되지 않아 학습을 종료합니다."
                    )
                    break

            scheduler.step(val_f1)

        fold_weight_paths.append(best_path)

    return fold_weight_paths


# -------------------------------------------------------------
# 7. 테스트 예측 및 제출 파일 생성
# -------------------------------------------------------------
def predict_with_ensemble(
    weight_paths: List[str],
    test_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> np.ndarray:
    """Fold별 모델을 로드해 소프트 앙상블한 후 최종 예측을 반환합니다."""

    all_fold_probs: List[np.ndarray] = []

    for fold_idx, weight_path in enumerate(weight_paths, start=1):
        print(f"\n[Inference] Fold {fold_idx} 가중치 로드: {weight_path}")
        model = timm.create_model(config.model_name, pretrained=False, num_classes=17).to(device)
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        fold_probs: List[np.ndarray] = []
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc=f"Predict Fold {fold_idx}", leave=False):
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                fold_probs.append(probs.detach().cpu().numpy())

        all_fold_probs.append(np.concatenate(fold_probs, axis=0))

    stacked_probs = np.stack(all_fold_probs, axis=0)  # (num_folds, N, num_classes)
    mean_probs = stacked_probs.mean(axis=0)
    final_preds = mean_probs.argmax(axis=1)

    print("✅ 앙상블 예측 완료!")
    return final_preds


def save_submission(
    test_dataset: DocumentDataset,
    predictions: np.ndarray,
    sample_submission_path: str,
    output_path: str,
) -> None:
    """예측 결과를 sample_submission과 동일한 포맷으로 저장합니다."""

    submission_df = pd.DataFrame({"ID": test_dataset.image_names, "target": predictions})
    sample_df = pd.read_csv(sample_submission_path)

    # ID 순서 검증: 실제 대회 제출 시 매우 중요합니다.
    assert (sample_df["ID"].values == submission_df["ID"].values).all(), "ID 순서가 일치하지 않습니다."

    submission_df.to_csv(output_path, index=False)
    print(f"✅ 제출 파일 저장 완료: {output_path}")


# -------------------------------------------------------------
# 8. 스크립트 진입점
# -------------------------------------------------------------
def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 장치(Device): {device}")

    # 경로 설정 (노트북과 동일하게 baseline 디렉터리 기준 상대경로를 사용합니다.)
    train_csv = "../data/train.csv"
    train_dir = "../data/train"
    aug_dir = "../data/train_aug"
    aug_csv = "../data/aug_train.csv"
    sample_csv = "../data/sample_submission.csv"
    test_dir = "../data/test"

    # 오프라인 증강 실행 (이미 생성되어 있다면 덮어쓰기)
    create_offline_augmented_dataset(train_csv, train_dir, aug_dir, aug_csv, num_augments=4)

    # 변환 및 데이터셋 준비
    config = TrainingConfig()
    train_transform, test_transform = build_transforms(config.image_size)

    train_dataset = DocumentDataset(aug_csv, aug_dir, transform=train_transform)
    test_dataset = DocumentDataset(sample_csv, test_dir, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # K-Fold 학습 및 앙상블 추론
    weight_paths = train_kfold(train_dataset, config, device)
    predictions = predict_with_ensemble(weight_paths, test_loader, config, device)

    # 제출 파일 저장
    save_submission(test_dataset, predictions, sample_csv, "submission.csv")


if __name__ == "__main__":
    main()