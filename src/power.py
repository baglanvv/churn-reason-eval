"""How much can a 48-row test set actually resolve?

A portfolio eval that reports 95.8% without a confidence interval is
overclaiming. This prints the Wilson interval for the observed score and the
smallest difference the set could detect against it, which is the number that
decides whether a prompt change is a real improvement or noise.

  python3 src/power.py
"""
import math


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def min_detectable_diff(p1, n, z=1.96, power_z=0.84):
    """Smallest p2 > p1 detectable at ~95% confidence, ~80% power, two
    independent samples of size n (unpaired - the conservative case)."""
    for step in range(1, 1000):
        p2 = p1 + step / 1000.0
        if p2 >= 1.0:
            return None
        se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
        if (p2 - p1) >= (z + power_z) * se:
            return p2
    return None


def n_per_arm(p1, p2, z=1.96, power_z=0.84):
    """Rows per arm needed to call a p1 -> p2 improvement at ~95% / ~80% power."""
    return ((z + power_z) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))) / ((p2 - p1) ** 2)


if __name__ == "__main__":
    for k, n, name in [(46, 48, "baseline, held-out test"),
                       (45, 48, "baseline, dev"),
                       (92, 96, "baseline, all rows")]:
        lo, hi = wilson(k, n)
        print(f"{name}: {k}/{n} = {k/n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"width {hi-lo:.3f}")

    print()
    p1 = 46 / 48
    mdd = min_detectable_diff(p1, 48)
    print(f"At {p1:.3f} on n=48, the smallest improvement this set can")
    print(f"distinguish from noise is ~{mdd:.3f}" if mdd else
          "no improvement is detectable - the baseline is too close to ceiling")
    print()
    for target in (0.90, 0.95):
        lo, hi = wilson(round(target * 48), 48)
        print(f"a model scoring {target:.2f} on n=48 -> CI [{lo:.3f}, {hi:.3f}]"
              f"  overlaps baseline CI: {hi >= wilson(46,48)[0]}")
    print()
    print("Rows per arm needed to resolve a real difference:")
    for a, b in [(0.95, 0.98), (0.95, 0.97), (0.95, 0.96)]:
        print(f"  {a:.2f} -> {b:.2f}  (+{(b-a)*100:.0f}pp):  n = "
              f"{math.ceil(n_per_arm(a, b)):>5} per arm")
    print()
    print("Implication: this set sizes a GO/NO-GO on gross failure, not a")
    print("prompt-vs-prompt ranking.")
