# Churn-reason classification: an eval that failed its own power check

An evaluation harness for an LLM feature in EdTech retention — and a worked
example of why the first number an eval produces is usually not the finding.

**Headline: I built this to decide whether an LLM should classify cancellation
reasons. It cannot answer that question, and the reason it cannot is the
interesting part.** A $0 keyword baseline scores 95.8% on the held-out half,
which saturates the test set and leaves no detectable room for a model to win.
That is a verdict on my test set, not a verdict on LLMs.

---

## The product question

An EdTech subscription asks "why are you cancelling?" on the way out. The free
text is read by nobody at volume. If each comment were reliably labeled, the
save-flow could route automatically:

| Reason | Action it triggers | Save potential |
|---|---|---|
| `price` | discount or cheaper tier | high |
| `content_gap` | curriculum review + alternative track | medium |
| `time` | **offer a pause instead of cancellation** | high |
| `goal_met` | ask for testimonial — **never discount** | none |
| `technical` | engineering escalation + service credit | medium |
| `support` | senior agent follow-up | medium |
| `competitor` | competitive intel | low |
| `unclear` | human review | unknown |

Full definitions and costs: [`data/taxonomy.json`](data/taxonomy.json).

A label that does not change what the business does is not worth predicting, so
every class maps to exactly one action.

## Why the headline metric is dollars, not accuracy

Accuracy treats every mistake as equally bad. These are not:

- Calling a **`goal_met`** learner **`price`** hands a discount to someone who
  already finished and was leaving happy. Pure margin loss, zero save.
- Calling a **`time`** learner **`unclear`** forfeits the cheapest save in the
  taxonomy — a pause costs ~$2 and is the likeliest of a weak set of offers
  to be accepted, because it asks for no new money.

So the harness reports **regret in dollars**: the gap between the best
achievable outcome for a case and what the prediction actually produced.

```
correct -> net = save_rate[true] * LTV - action_cost[true]
wrong   -> net = -action_cost[pred]      (wrong remedy, save collapses, still billed)
regret  -> best_possible_net[true] - actual_net
```

A model can gain accuracy while losing money. Regret catches that; accuracy
does not. Assumptions live in [`src/score.py`](src/score.py) under
`ECONOMICS` — `LTV = $480`, per-class save rates — and are **planning
assumptions, not measurements.** Every dollar figure below inherits that
caveat.

An earlier version of this file used save rates of 0.35–0.55. Propagated
through the business model in the companion PRD, those implied a 12.7% overall
save rate and a $60k/month net — a fantasy, since published save-flow
conversion is single-digit to low-double-digit. The current rates are
deliberately pessimistic. The lesson generalises: **an eval's cost model is a
load-bearing assumption and deserves the same scrutiny as its labels.**

## Test set

96 rows, [`data/labeled_set.jsonl`](data/labeled_set.jsonl).

| | |
|---|---|
| Classes | 8, exactly 12 rows each |
| Split | 48 dev / 48 test, stratified — 6 per class, 16 hard, 9 RU on each side |
| Difficulty | 64 `easy`, 32 `hard` (hard = resolvable only by reading past keywords) |
| Language | 78 EN, 18 RU — the CIS market this is modeled on is bilingual |

**The data is synthetic. I wrote all 96 rows by hand.** It is not customer
data, and that is this project's central weakness, discussed below.

`hard` rows are where the design work went. Examples:

- *"Found the same SQL material on another platform for a third of the price."*
  → `price`, not `competitor`. The decisive factor is cost.
- *"I finished the only track relevant to me in three weeks and now there is
  nothing left to watch."* → `content_gap`, not `goal_met`. Sounds satisfied;
  the driver is a thin library.
- *"It was not for me."* → `unclear`. Abstaining is the correct answer, and a
  classifier that guesses here sends a real intervention to the wrong person.

## Results

Held-out test half, n=48. **Only the baseline row is measured.** I have no API
key configured, so the model rows are commands to run, not results — I am not
reporting numbers I did not produce.

| Run | Accuracy | 95% CI | Macro F1 | Hard | RU | Regret/case | Value captured |
|---|---|---|---|---|---|---|---|
| keyword baseline | **0.958** | [0.860, 0.988] | 0.959 | 0.875 | 1.000 | $0.51 | 96.7% |
| `v1` minimal prompt | — | — | — | — | — | — | — |
| `v2` + definitions | — | — | — | — | — | — | — |
| `v3` + actions + 5 dev few-shots | — | — | — | — | — | — | — |

Baseline per-class (test): perfect precision on 7 of 8 classes; `unclear`
precision 0.75 because it absorbs what the rules cannot match. Both test
errors are `hard` rows falling into `unclear`:

```
p08: price    -> unclear   "only needed two courses, paying for the whole library makes no sense"
g10: goal_met -> unclear   "would keep paying but there is nothing more I need"
```

That `price -> unclear` single error carries **$17.60 of the $24.60 total
regret** — 72% of the cost sits in one row, which is what a per-case average hides.
Reproduce: `results/baseline-keywords.json`.

## What went wrong with the method

Two defects, both found by instrumenting against them. Neither is patched out
of the repo, because they are the result.

**1. The baseline is contaminated and its score is a ceiling, not an estimate.**
I wrote the keyword rules while looking at all 96 rows, *before* the dev/test
split existed. Splitting afterwards does not decontaminate anything. The
evidence is in the numbers: test (0.958) scores *higher* than dev (0.938),
which is the signature of leakage across both halves rather than overfitting to
one. The consequence for the comparison is directional and worth stating
plainly: **the LLM rows will be genuinely zero-shot while the baseline is not,
so the comparison is biased in the baseline's favour.** A model that merely
ties the baseline is in reality better.

**2. The set is too small to detect the improvement it was built to measure.**
`python3 src/power.py`:

```
baseline, held-out test: 46/48 = 0.958  95% CI [0.860, 0.988]  width 0.128

At 0.958 on n=48, no improvement is detectable — the baseline is
too close to ceiling.

a model scoring 0.90 on n=48 -> CI [0.778, 0.955]  overlaps baseline CI: True
a model scoring 0.95 on n=48 -> CI [0.860, 0.988]  overlaps baseline CI: True
```

Any LLM result I now generate would be statistically indistinguishable from the
baseline. Running the model rows would produce a table that *looks* like
evidence and is not. **This set sizes a go/no-go on gross failure. It cannot
rank prompts.**

A guard against the leak I actually made is in the repo —
[`src/check_contamination.py`](src/check_contamination.py). The first draft of
`prompts/v3.txt` had 3 of 4 few-shot examples drawn from the test half. The
guard caught it, the examples were replaced with dev-only rows, and it now runs
clean and fails on both leak shapes.

## What I would do before trusting any number here

1. **Replace the synthetic data with real cancellation comments.** A baseline
   saturating at 95.8% is evidence my invented text is too clean and too
   on-distribution for rules I wrote myself — real comments are messier,
   multi-reason, typo-ridden, and code-switched mid-sentence. This is the only
   fix that matters; everything else is secondary.
2. **Size the set from the decision.** `src/power.py` prints the arithmetic:
   resolving a 3-point gain near 95% needs ~585 rows per arm, a 2-point gain
   ~1,500, a 1-point gain ~6,700. Label to the decision, not to a round number.
3. **Have someone else label the held-out half**, so rules and data do not share
   an author.
4. **Measure the save rates.** Every dollar figure rests on assumed retention
   rates. Two weeks of a real save-flow A/B replaces the whole guess.
5. **Only then compare prompts**, and only on cost-weighted regret.

The honest recommendation today: **ship the keyword baseline behind a confidence
threshold, route everything it cannot match to human review, and use the
resulting human labels as the real test set.** The baseline is free, auditable,
and good enough to be a data-collection instrument. Buy the LLM decision with
real data instead of guessing at it.

## Running it

No dependencies for the baseline — stdlib Python 3.9+.

```bash
python3 src/run_eval.py --model baseline      # measured above
python3 src/check_contamination.py            # leak guard, exit 1 on failure
python3 src/power.py                          # confidence intervals
```

For the model rows:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
python3 src/run_eval.py --model claude-sonnet-5 --prompt prompts/v3.txt
```

Off-taxonomy responses are counted and reported as `off_taxonomy_responses`
rather than silently repaired — a model inventing a label is a real failure
mode and belongs in the results.

```
data/    taxonomy.json (classes, actions, costs) · labeled_set.jsonl (96 rows)
prompts/ v1 minimal · v2 + definitions · v3 + actions + dev-only few-shots
src/     run_eval.py · score.py (regret model) · baseline.py · power.py · check_contamination.py
results/ baseline-keywords.json
```
