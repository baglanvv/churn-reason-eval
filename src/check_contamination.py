"""Guard: no prompt may contain a held-out test example.

Few-shot examples must come from the dev half only. Pasting a test row into a
prompt inflates the score silently and the resulting number is worthless. This
caught a real leak during development (3 of 4 examples in the first draft of
prompts/v3.txt were test rows), which is why it lives in the repo.

  python3 src/check_contamination.py   -> exit 1 on any leak
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Below this length a row's text is generic enough to appear inside ordinary
# prompt prose, so substring matching is not evidence of a leak.
MIN_SUBSTRING_LEN = 25


def main():
    with open(os.path.join(ROOT, "data", "labeled_set.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    test_texts = {r["text"]: r["id"] for r in rows if r["split"] == "test"}
    by_text = {r["text"]: r for r in rows}

    leaks = []
    checked = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "prompts", "*.txt"))):
        body = open(path).read()
        rel = os.path.relpath(path, ROOT)

        # Pass 1 - any row used as a labeled few-shot example must be dev.
        # Length does not matter here: the quoted form is unambiguous.
        for text, label in re.findall(r'Comment: "(.+?)"\nLabel: (\w+)', body):
            checked += 1
            r = by_text.get(text)
            if r is None:
                continue
            if r["split"] == "test":
                leaks.append(f"{rel}: few-shot example {r['id']} is a TEST row")
            if r["label"] != label:
                leaks.append(
                    f"{rel}: example {r['id']} labeled '{label}' but truth is '{r['label']}'")

        # Pass 2 - a test row pasted in loosely, outside the example format.
        # Only scan rows long enough to be unambiguous: short labels like
        # "cancel" or "ok" occur inside ordinary prose ("cancellation",
        # "look for") and produced false positives when this scanned everything.
        for text, rid in test_texts.items():
            if len(text) >= MIN_SUBSTRING_LEN and text in body:
                leaks.append(f"{rel}: contains test row {rid} verbatim")

    if leaks:
        print("FAIL")
        for l in leaks:
            print("  " + l)
        return 1
    print(f"PASS - {checked} labeled examples across "
          f"{len(glob.glob(os.path.join(ROOT, 'prompts', '*.txt')))} prompts, "
          f"no test rows present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
