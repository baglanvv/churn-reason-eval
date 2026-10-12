"""Run one classifier over the labeled set and write a scored result file.

  python3 src/run_eval.py --model baseline
  python3 src/run_eval.py --model claude-sonnet-5 --prompt prompts/v1.txt

The baseline needs nothing installed. The Claude runs need:
  pip install anthropic   and   export ANTHROPIC_API_KEY=...
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import score as scoring  # noqa: E402
import baseline  # noqa: E402


def load_rows():
    path = os.path.join(ROOT, "data", "labeled_set.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def predict_llm(rows, model, prompt_path):
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic not installed. Run: pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")

    with open(prompt_path) as f:
        template = f.read()

    client = anthropic.Anthropic()
    valid = set(scoring.CLASSES)
    preds, raw = [], []

    for i, r in enumerate(rows, 1):
        msg = client.messages.create(
            model=model,
            max_tokens=16,
            temperature=0,
            messages=[{"role": "user",
                       "content": template.replace("{{TEXT}}", r["text"])}],
        )
        out = msg.content[0].text.strip().lower()
        raw.append(out)
        # An off-taxonomy answer is a real failure mode, not something to
        # silently repair. Count it as unclear and report how often it happens.
        preds.append(out if out in valid else "unclear")
        print(f"\r  {i}/{len(rows)}", end="", flush=True)
    print()

    off_taxonomy = sum(1 for o in raw if o not in valid)
    return preds, {"off_taxonomy_responses": off_taxonomy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="'baseline' or a Claude model id e.g. claude-sonnet-5")
    ap.add_argument("--prompt", default="prompts/v1.txt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows()
    extra = {}

    if args.model == "baseline":
        preds = [baseline.classify(r["text"]) for r in rows]
        run_name = "baseline-keywords"
    else:
        preds, extra = predict_llm(
            rows, args.model, os.path.join(ROOT, args.prompt))
        run_name = f"{args.model}+{os.path.basename(args.prompt)}"

    for r, p in zip(rows, preds):
        r["pred"] = p

    test_rows = [r for r in rows if r["split"] == "test"]
    dev_rows = [r for r in rows if r["split"] == "dev"]

    result = {
        "run": run_name,
        # Headline is the HELD-OUT half. The keyword rules were authored while
        # looking at the dev half, so dev numbers are contaminated for the
        # baseline and are kept only to show the size of that gap.
        "metrics": scoring.score(test_rows),
        "metrics_dev": scoring.score(dev_rows),
        "metrics_all": scoring.score(rows),
        "costliest_confusions": scoring.top_confusions(test_rows),
        "economics": scoring.ECONOMICS,
        "notes": extra,
        "predictions": [
            {"id": r["id"], "split": r["split"], "label": r["label"],
             "pred": r["pred"], "difficulty": r["difficulty"], "lang": r["lang"]}
            for r in rows
        ],
    }

    out = args.out or os.path.join(
        ROOT, "results", run_name.replace("/", "_") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    m = result["metrics"]
    d = result["metrics_dev"]
    print(f"\nrun: {run_name}")
    print(f"  --- HELD-OUT TEST (n={m['n']}) ---")
    print(f"  accuracy        {m['accuracy']:.3f}   (easy {m['accuracy_easy']} / hard {m['accuracy_hard']})")
    print(f"  macro F1        {m['macro_f1']:.3f}")
    print(f"  EN / RU         {m['accuracy_en']} / {m['accuracy_ru']}")
    print(f"  regret          ${m['regret_usd_total']:.0f} total, ${m['regret_usd_per_case']:.2f}/case")
    print(f"  value captured  {m['value_captured_pct']}%")
    print(f"  dev accuracy    {d['accuracy']:.3f}   <- contaminated for baseline; gap vs test = overfit")
    print(f"\n  costliest confusions (test):")
    for c in result["costliest_confusions"]:
        print(f"    {c['confusion']:<28} n={c['count']}  ${c['regret_usd']:.0f}")
    print(f"\nwrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
