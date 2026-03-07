"""
Phrase Boundary Prediction — Evaluation Metrics

Usage:
    python src/metrics.py <predictions.jsonl>

Input JSONL format (one JSON object per line):
    {
        "user":       "A404 G404 C504 ...",          # 原始音符序列（無 |）
        "gt":         "A404 G404 C504 a402 | ...",   # Ground truth（含 |）
        "prediction": "A404 G404 C504 a402 | ...",   # 模型預測（含 |）
        ...                                          # 其他 metadata 欄位（optional）
    }

Metrics computed:
    1. Boundary-level: Precision / Recall / F1
    2. Note-level: accuracy (ignoring '|')
    3. Sequence-level: exact match ratio
"""

import json
import re
import sys


# ─── Token Parsing ───────────────────────────────────────────────

def clean_and_split_tokens(s: str) -> list[str]:
    """
    清理特殊 token 並以空白切分。
    去掉 <end_of_turn>、<start_of_turn> 等標記。
    """
    if s is None:
        return []
    s = s.replace("<end_of_turn>", " ").replace("<start_of_turn>", " ")
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return s.split(" ")


def extract_notes_and_boundaries(tokens: list[str]) -> tuple[list[str], set[int]]:
    """
    從 token 列表中分離音符與邊界位置。

    Args:
        tokens: 含有音符和 '|' 的 token 列表

    Returns:
        notes:      只含音符的列表（去掉所有 '|'）
        boundaries: 邊界位置集合（以音符索引表示）
                    例如 '|' 出現在第 i 個音符之後 → boundary index = i
    """
    notes = []
    boundaries = set()
    note_idx = -1
    for tok in tokens:
        if tok == "|":
            if note_idx >= 0:
                boundaries.add(note_idx)
            continue
        notes.append(tok)
        note_idx += 1
    return notes, boundaries


# ─── Metrics Computation ─────────────────────────────────────────

def compute_metrics(records: list[dict]) -> dict:
    """
    計算所有評估指標。

    Args:
        records: 每筆包含 'user', 'gt', 'prediction' 的 dict

    Returns:
        dict with all computed metrics
    """
    tp = fp = fn = 0
    total_notes = 0
    note_matches = 0
    exact_seq = 0
    mismatch_cases = []

    for rec in records:
        user = rec["user"]
        gt = rec["gt"]
        pred = rec["prediction"]

        user_tokens = clean_and_split_tokens(user)
        gt_tokens = clean_and_split_tokens(gt)
        pred_tokens = clean_and_split_tokens(pred)

        user_notes, _ = extract_notes_and_boundaries(user_tokens)
        gt_notes, gt_b = extract_notes_and_boundaries(gt_tokens)
        pred_notes, pr_b = extract_notes_and_boundaries(pred_tokens)

        # Note-level consistency
        # Denominator = len(user_notes): the task constraint is "don't modify
        # the original sequence", so we measure against the user's input length.
        # If pred is shorter (model stopped early), the missing notes count as
        # mismatches. If pred is longer, extra notes are ignored (not penalized
        # in the denominator) — this is conservative but matches the task spec.
        L = min(len(user_notes), len(pred_notes))
        total_notes += len(user_notes)
        note_matches += sum(1 for i in range(L) if user_notes[i] == pred_notes[i])

        # Sequence-level exact match
        if len(user_notes) == len(pred_notes) and all(
            user_notes[i] == pred_notes[i] for i in range(len(user_notes))
        ):
            exact_seq += 1
        else:
            mismatch_cases.append({
                "index": rec.get("index"),
                "bwv_number": rec.get("bwv_number"),
                "title": rec.get("title"),
                "len_user": len(user_notes),
                "len_pred": len(pred_notes),
            })

        # Boundary-level metrics (set-based exact match in note-index space)
        # This works even when pred length differs from gt, because boundaries
        # are represented as note indices — any index mismatch is a FP or FN.
        tp += len(gt_b & pr_b)
        fp += len(pr_b - gt_b)
        fn += len(gt_b - pr_b)

    total_samples = len(records)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    note_acc = note_matches / total_notes if total_notes > 0 else 0.0
    exact_ratio = exact_seq / total_samples if total_samples > 0 else 0.0

    return {
        "total_samples": total_samples,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_notes": total_notes,
        "note_matches": note_matches,
        "note_accuracy": note_acc,
        "exact_seq": exact_seq,
        "exact_ratio": exact_ratio,
        "mismatch_cases": mismatch_cases,
    }


# ─── Output Formatting ───────────────────────────────────────────

def print_metrics(m: dict) -> None:
    """格式化輸出所有指標。"""
    print(f"Total test samples: {m['total_samples']}")

    print("=== Boundary-level (note index, exact match) ===")
    print(f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}")

    print()
    print("=== Note-level consistency ===")
    print(f"Total notes = {m['total_notes']}   Matches = {m['note_matches']}")
    print(f"Note accuracy = {m['note_accuracy']:.4f}")

    print()
    print("=== Sequence-level exact match ===")
    print(f"Exact = {m['exact_seq']}/{m['total_samples']}   Ratio = {m['exact_ratio']:.4f}")

    mismatches = m["mismatch_cases"]
    print(f"\nMismatch: {len(mismatches)} sample(s)")
    for mc in mismatches:
        idx = mc.get("index", "?")
        bwv = mc.get("bwv_number", "?")
        title = mc.get("title", "")
        print(f'  [{idx}] BWV {bwv} "{title}" (user={mc["len_user"]}, pred={mc["len_pred"]})')


# ─── Main ─────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/metrics.py <predictions.jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    m = compute_metrics(records)
    print_metrics(m)


if __name__ == "__main__":
    main()
