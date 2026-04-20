import json
from collections import Counter
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FINAL_SCORE_OUTPUT

# =========================
# Command line arguments
# =========================
parser = argparse.ArgumentParser(description='Analyze score distribution')
parser.add_argument('--input', help='Input JSON file')
args = parser.parse_args()

INPUT_FILE = args.input if args.input else FINAL_SCORE_OUTPUT

def main():
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {INPUT_FILE}: {e}")
        return

    if not isinstance(data, list) or len(data) == 0:
        print("[WARN] Input file is empty or not a list.")
        return

    total = 0
    counter = Counter()

    for item in data:
        score = item.get("score")
        if isinstance(score, int) and 0 <= score <= 4:
            counter[score] += 1
            total += 1

    if total == 0:
        print("[WARN] No valid scores found.")
        return

    print("=== Final Score Distribution Statistics (0–4) ===")
    print(f"{'Score':>4} {'Count':>8} {'Percentage':>10}")
    print("-" * 26)

    for s in range(5):
        cnt = counter.get(s, 0)
        ratio = cnt / total * 100
        print(f"{s:>4} {cnt:>8} {ratio:>9.2f}%")

    print("-" * 26)
    print(f"Total samples: {total}")


if __name__ == "__main__":
    main()

