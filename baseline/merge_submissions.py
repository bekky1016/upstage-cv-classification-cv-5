# -*- coding: utf-8 -*-
# filename: merge_submissions.py

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# 공통 유틸
# -----------------------------
def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: str(c).strip())

def normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """id/ID/filename 등 다양한 id 열을 'id'로 통일"""
    df = _strip_cols(df)
    lower = {c.lower(): c for c in df.columns}
    for cand in ["id", "ID", "Id", "file_name", "filename", "image", "img", "name"]:
        if cand in df.columns:
            if cand != "id":
                df = df.rename(columns={cand: "id"})
            return df
    for k, v in lower.items():
        if k in {"id", "file_name", "filename", "image", "img", "name"}:
            if v != "id":
                df = df.rename(columns={v: "id"})
            return df
    first = df.columns[0]
    if first != "id":
        df = df.rename(columns={first: "id"})
    return df

def find_label_col(df: pd.DataFrame, prefer=("class","target","label","prediction","pred","category")) -> str | None:
    """라벨/클래스 컬럼 자동 탐지 (id 제외 후 후보 우선)"""
    cols = [c for c in df.columns if c != "id"]
    lower = {c.lower(): c for c in cols}
    for cand in prefer:
        if cand in lower:
            return lower[cand]
    if len(cols) == 1:
        return cols[0]
    return None

def is_label_submission(df: pd.DataFrame) -> bool:
    """`id` + 라벨 하나(2열)"""
    return "id" in [c.lower() for c in df.columns] and len(df.columns) == 2

def is_prob_submission(df: pd.DataFrame) -> bool:
    """`id` + 여러 확률 열"""
    return "id" in [c.lower() for c in df.columns] and len(df.columns) > 2

def parse_parent_from_subclass_col(cname: str):
    """
    서브클래스 확률 열 이름에서 부모 클래스 인덱스 유추
    예) 'sc31' -> 3, '31' -> 3, 'c141' -> 14
    """
    s = str(cname).lower()
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    v = int(m.group(1))
    return v // 10 if v >= 10 else v

def to_parent_label(x):
    """라벨(예: 31, 72, 141)을 부모(3,7,14)로 변환. 변환 불가 시 None"""
    try:
        xi = int(x)
        return xi // 10 if xi >= 10 else xi
    except Exception:
        return None

def align_by_id(ids_base, arr_base, ids_other, arr_other):
    """두 배열을 공통 id 순서로 정렬/정합"""
    a = pd.DataFrame({"id": ids_base, "_i": np.arange(len(ids_base))}).set_index("id")
    b = pd.DataFrame({"id": ids_other, "_j": np.arange(len(ids_other))}).set_index("id")
    inter = a.join(b, how="inner")
    if inter.empty:
        raise ValueError("두 파일의 id가 겹치지 않습니다.")
    idx_a = inter["_i"].values
    idx_b = inter["_j"].values
    return inter.index.values, arr_base[idx_a], arr_other[idx_b]


# -----------------------------
# 라벨형 머지
# -----------------------------
def merge_label_mode(sota_df: pd.DataFrame, sub_df: pd.DataFrame, target=(3, 7, 14)) -> pd.DataFrame:
    """
    sota_df: id + (class/target/label/...)
    sub_df : id + (subclass or target or class)
      - sub_df의 target이 '서브클래스 코드(31/71/141...)'여도 자동으로 부모(3/7/14)로 변환
    - SOTA가 target으로 예측한 행만 서브 결과로 교체
    """
    sota = normalize_id_column(sota_df.copy())
    sub  = normalize_id_column(sub_df.copy())

    # SOTA 라벨 컬럼 → 'class'
    sota_label = find_label_col(sota)
    if sota_label is None:
        raise ValueError(f"SOTA CSV에서 라벨 컬럼을 찾지 못했습니다. 열 목록: {list(sota.columns)}")
    if sota_label != "class":
        sota = sota.rename(columns={sota_label: "class"})

    # SUB 라벨/서브클래스 컬럼 찾기
    cols_lower = {c.lower(): c for c in sub.columns}
    sub_col = None
    # 우선순위: subclass > target > class
    for key in ("subclass", "target", "class"):
        if key in cols_lower:
            sub_col = cols_lower[key]
            break
    if sub_col is None:
        # 자동 탐지
        sub_label_auto = find_label_col(sub)
        if sub_label_auto is None:
            raise ValueError(f"서브 CSV에서 'subclass/target/class' 또는 라벨 컬럼을 찾지 못했습니다. 열 목록: {list(sub.columns)}")
        sub_col = sub_label_auto

    # sub_col이 서브클래스 코드일 수도(31/71/141...), 부모로 환산
    sub["class_sub"] = sub[sub_col].map(to_parent_label)

    out = sota.merge(sub[["id", "class_sub"]], on="id", how="left")
    mask = out["class"].isin(target) & out["class_sub"].notna()
    out.loc[mask, "class"] = out.loc[mask, "class_sub"].astype(int)

    # 최종 출력은 대회 포맷: ID, target
    out = out.rename(columns={"id": "ID", "class": "target"})
    # 정수 캐스팅(가능하면)
    try:
        out["target"] = out["target"].astype(int)
    except Exception:
        pass
    return out[["ID", "target"]]


# -----------------------------
# 확률형 머지
# -----------------------------
def df_to_prob_matrix(df: pd.DataFrame):
    """
    확률형 제출을 행렬로 변환.
    - 열 이름: c0.. / 0.. / class_0.. 등 → 숫자만 추출
    """
    df = normalize_id_column(df.copy())
    class_cols = [c for c in df.columns if c.lower() != "id"]
    pairs = []
    for c in class_cols:
        s = str(c).lower()
        m = re.search(r"(\d+)", s)
        if not m:
            raise ValueError(f"확률 열 이름에서 번호를 추출할 수 없습니다: {c}")
        pairs.append((int(m.group(1)), c))
    pairs.sort(key=lambda x: x[0])
    ordered_cols = [c for _, c in pairs]
    ids = df["id"].values
    probs = df[ordered_cols].values.astype(float)
    return ids, probs, ordered_cols

def subclass_probs_to_parent(df: pd.DataFrame, num_parent: int):
    """서브클래스 확률 → 부모 클래스 확률 합산"""
    df = normalize_id_column(df.copy())
    ids = df["id"].values
    p_parent = np.zeros((len(df), num_parent), dtype=float)
    for c in df.columns:
        if c == "id":
            continue
        p = parse_parent_from_subclass_col(c)
        if p is None or p < 0 or p >= num_parent:
            continue
        p_parent[:, p] += df[c].values.astype(float)
    s = p_parent.sum(axis=1, keepdims=True)
    nz = s > 0
    p_parent[nz] /= s[nz]
    return ids, p_parent

def merge_prob_mode(
    sota_prob_df: pd.DataFrame,
    sub_prob_df: pd.DataFrame,
    target=(3, 7, 14),
    w_sota=1.0,
    w_sub=2.0,
    margin=0.0,
) -> pd.DataFrame:
    """
    - SOTA 확률: id + (c0.. 혹은 0.. 등)
    - Sub 확률: id + (31, sc31, ...) → 부모로 합산
    - target에서만 결합/교체
    - margin>0이면: p_sub_parent[c] >= p_sota[c] * (1+margin)일 때만 교체
      margin=0이면: 가중 평균 결합
    """
    ids_sota, p_sota, _ = df_to_prob_matrix(sota_prob_df)
    C = p_sota.shape[1]
    ids_sub, p_sub_parent = subclass_probs_to_parent(sub_prob_df, C)

    # id 정렬/정합
    ids, p_sota, p_sub_parent = align_by_id(ids_sota, p_sota, ids_sub, p_sub_parent)

    pred_sota = p_sota.argmax(axis=1)
    mask = np.isin(pred_sota, list(target))[:, None]  # [N,1] bool

    if margin > 0:
        p_final = p_sota.copy()
        for c in target:
            cond = (p_sub_parent[:, c] >= p_sota[:, c] * (1.0 + margin)) & mask.squeeze()
            p_final[cond, c] = p_sub_parent[cond, c]
        pred_final = p_final.argmax(axis=1)
    else:
        w_s = np.ones((1, C)) * w_sota
        w_b = np.ones((1, C)) * w_sub
        for c in range(C):
            if c not in target:
                w_b[0, c] = 0.0
        p_final = (p_sota * (1 - mask)) + mask * ((p_sota * w_s + p_sub_parent * w_b) / (w_s + w_b + 1e-12))
        pred_final = p_final.argmax(axis=1)

    # 최종 출력은 대회 포맷: ID, target
    out = pd.DataFrame({"ID": ids, "target": pred_final})
    try:
        out["target"] = out["target"].astype(int)
    except Exception:
        pass
    return out[["ID", "target"]]


# -----------------------------
# 엔트리
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="SOTA 결과와 Sub-class 결과를 결합하여 3/7/14만 교정 (출력: ID,target)")
    parser.add_argument("--sota", required=True, help="SOTA CSV 경로 (라벨형: id/class/target.. / 확률형: id,c0..)")
    parser.add_argument("--sub", required=True, help="서브클래스 CSV 경로 (라벨형: ID/subclass/target.. / 확률형: id,31/sc31..)")
    parser.add_argument("--out", required=True, help="저장 경로")
    parser.add_argument("--target", default="3,7,14", help="보정 대상 클래스 (쉼표 구분)")
    parser.add_argument("--mode", choices=["auto", "label", "prob"], default="auto", help="머지 모드")
    parser.add_argument("--w_sub", type=float, default=2.0, help="확률 결합 시 서브 가중치")
    parser.add_argument("--w_sota", type=float, default=1.0, help="확률 결합 시 SOTA 가중치")
    parser.add_argument("--margin", type=float, default=0.0, help="확률 모드에서 교체 마진(>0이면 하드 교체)")
    args = parser.parse_args()

    target = tuple(int(x) for x in args.target.split(","))

    sota = pd.read_csv(args.sota)
    sub  = pd.read_csv(args.sub)

    mode = args.mode
    if mode == "auto":
        if is_label_submission(sota) and (is_label_submission(sub) or any(k in [c.lower() for c in sub.columns] for k in ["subclass","target","class"])):
            mode = "label"
        elif is_prob_submission(sota) and is_prob_submission(sub):
            mode = "prob"
        else:
            mode = "label"

    print(f"▶ Merge mode: {mode}")
    if mode == "label":
        out = merge_label_mode(sota, sub, target=target)
    else:
        out = merge_prob_mode(
            sota_prob_df=sota,
            sub_prob_df=sub,
            target=target,
            w_sota=args.w_sota,
            w_sub=args.w_sub,
            margin=args.margin,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"✅ Saved: {args.out} (rows={len(out)})")


if __name__ == "__main__":
    main()
