"""Scoring for churn-reason classification.

Accuracy is the wrong headline metric here. Each predicted class triggers a
different retention action with a different cost, and each true class has a
different chance of being saved if handled correctly. A model can gain accuracy
while losing money by trading cheap high-value saves for expensive wrong ones.

So we report dollar REGRET alongside the usual classification metrics.

Economic model (all assumptions explicit and editable in ECONOMICS):
  correct prediction -> net = save_rate[true] * LTV - action_cost[true]
  wrong prediction   -> net = -action_cost[pred]
                        (the action does not address the real reason, so the
                         save probability collapses, and you still pay for it)
  regret             -> best_possible_net[true] - actual_net

Stdlib only, no dependencies.
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# --- Assumptions. Change these to match your own unit economics. -------------
LTV = 480.0  # 12 months at $40. The value of a save.

# Probability a churning learner is actually retained when the CORRECT action
# is applied. Published save-flow conversion is single-digit to low-double-
# digit. An earlier version of this file used 0.35-0.55, which is not
# defensible: propagated through the business model in the companion PRD it
# implied a 12.7% overall save rate and a $60k/mo net. These are deliberately
# pessimistic, and they are planning assumptions, not measurements.
SAVE_RATE = {
    "price": 0.12,
    "content_gap": 0.04,  # "try another course" after "your courses were bad"
    "time": 0.18,         # a pause asks for no new money - best of a weak set
    "goal_met": 0.00,     # nothing to save, they are finished
    "technical": 0.08,
    "support": 0.10,
    "competitor": 0.02,
    "unclear": 0.03,      # after human review
}
# -----------------------------------------------------------------------------

with open(os.path.join(ROOT, "data", "taxonomy.json")) as f:
    TAXONOMY = json.load(f)["classes"]

CLASSES = list(TAXONOMY.keys())
ACTION_COST = {c: TAXONOMY[c]["action_cost_usd"] for c in CLASSES}

ECONOMICS = {"ltv": LTV, "save_rate": SAVE_RATE, "action_cost": ACTION_COST}


def best_possible_net(true_label):
    return SAVE_RATE[true_label] * LTV - ACTION_COST[true_label]


def actual_net(true_label, pred_label):
    if pred_label == true_label:
        return SAVE_RATE[true_label] * LTV - ACTION_COST[true_label]
    return -ACTION_COST.get(pred_label, 0.0)


def regret(true_label, pred_label):
    return best_possible_net(true_label) - actual_net(true_label, pred_label)


def score(rows):
    """rows: list of dicts with keys: label (truth), pred, difficulty, lang."""
    n = len(rows)
    correct = sum(1 for r in rows if r["pred"] == r["label"])

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    confusion = defaultdict(int)

    total_regret = 0.0
    for r in rows:
        t, p = r["label"], r["pred"]
        confusion[(t, p)] += 1
        total_regret += regret(t, p)
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1

    per_class = {}
    f1s = []
    for c in CLASSES:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "support": tp[c] + fn[c],
        }
        f1s.append(f1)

    def subset_acc(pred):
        sub = [r for r in rows if pred(r)]
        if not sub:
            return None
        return round(sum(1 for r in sub if r["pred"] == r["label"]) / len(sub), 3)

    # Ceiling: what a perfect classifier would net. Floor: predict the single
    # most common class for everything, the "do not bother with ML" option.
    ceiling = sum(best_possible_net(r["label"]) for r in rows)

    return {
        "n": n,
        "accuracy": round(correct / n, 3),
        "macro_f1": round(sum(f1s) / len(f1s), 3),
        "accuracy_easy": subset_acc(lambda r: r["difficulty"] == "easy"),
        "accuracy_hard": subset_acc(lambda r: r["difficulty"] == "hard"),
        "accuracy_en": subset_acc(lambda r: r["lang"] == "en"),
        "accuracy_ru": subset_acc(lambda r: r["lang"] == "ru"),
        "regret_usd_total": round(total_regret, 2),
        "regret_usd_per_case": round(total_regret / n, 2),
        "value_captured_pct": round(100.0 * (ceiling - total_regret) / ceiling, 1)
        if ceiling else None,
        "per_class": per_class,
        "confusion": {f"{t}->{p}": v for (t, p), v in sorted(confusion.items())},
    }


def top_confusions(rows, k=5):
    """The confusions that cost the most money, which is not the same as the
    confusions that happen most often."""
    agg = defaultdict(lambda: {"count": 0, "regret": 0.0})
    for r in rows:
        if r["pred"] != r["label"]:
            key = f"{r['label']} -> {r['pred']}"
            agg[key]["count"] += 1
            agg[key]["regret"] += regret(r["label"], r["pred"])
    ranked = sorted(agg.items(), key=lambda kv: -kv[1]["regret"])
    return [
        {"confusion": k_, "count": v["count"], "regret_usd": round(v["regret"], 2)}
        for k_, v in ranked[:k]
    ]
