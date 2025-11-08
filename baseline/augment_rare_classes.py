# ============================================================
# 🔹 오프라인 복제 (단일 폴더 구조 유지)
#    - 부족한 클래스(1,13,14)만 복제해서 총 100장씩
#    - train_balanced 폴더 안에는 모든 이미지가 평면 구조로 저장됨
# ============================================================
import os
import shutil
import random
import pandas as pd
from tqdm import tqdm

train_dir = "../data/train/"
balanced_dir = "../data/train_balanced/"
os.makedirs(balanced_dir, exist_ok=True)

# train.csv 로드
train_df = pd.read_csv("../data/train.csv")

# 클래스별 개수 확인
class_counts = train_df["target"].value_counts().to_dict()
target_per_class = 100
augment_classes = [1, 13, 14]

# 새로 저장할 balanced_df
balanced_records = []

print("▶ 복제 시작")
for cls in sorted(train_df["target"].unique()):
    cls_df = train_df[train_df["target"] == cls]

    # 원본 먼저 복사
    for _, row in cls_df.iterrows():
        src = os.path.join(train_dir, row["ID"])
        dst = os.path.join(balanced_dir, row["ID"])
        shutil.copy(src, dst)
        balanced_records.append({"ID": row["ID"], "target": row["target"]})  # ✅ 수정됨

    # 부족한 클래스만 복제
    if cls in augment_classes:
        current_n = len(cls_df)
        need_n = target_per_class - current_n
        print(f" - 클래스 {cls}: {current_n} → {target_per_class} (복제 {need_n}장)")

        img_list = cls_df["ID"].tolist()
        for i in tqdm(range(need_n)):
            src_file = random.choice(img_list)
            src_path = os.path.join(train_dir, src_file)
            new_name = f"copy_{i:03d}_{src_file}"
            dst_path = os.path.join(balanced_dir, new_name)
            shutil.copy(src_path, dst_path)
            balanced_records.append({"ID": new_name, "target": cls})  # ✅ dict 형태 통일

# 새 csv 저장
balanced_df = pd.DataFrame(balanced_records)
balanced_df.to_csv("../data/train_balanced.csv", index=False)
print("✅ 복제 완료: '../data/train_balanced/' 에 모든 이미지 저장됨")
print("✅ 메타데이터 저장: '../data/train_balanced.csv'")
