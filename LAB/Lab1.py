import sys
import os
import math
from collections import Counter
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

# 1) Вхідні дані (А, Б, В, Г, Д) - ВАРІАНТ 3
A = [
    [55, 60, 42, 43, 65], [30, 47, 47, 41, 52], [49, 44, 57, 61, 54],
    [50, 47, 57, 52, 40], [69, 47, 50, 58, 58], [42, 55, 51, 53, 58],
    [41, 30, 48, 54, 46], [50, 49, 62, 34, 35], [62, 41, 40, 38, 34],
    [63, 24, 41, 41, 46], [61, 64, 47, 54, 63], [53, 62, 56, 52, 51],
    [64, 36, 43, 52, 49], [47, 40, 35, 61, 38], [40, 55, 49, 62, 64]
]

B = [
    [48, 29, 52, 55, 63], [67, 64, 44, 53, 69], [49, 63, 53, 40, 45],
    [66, 41, 43, 60, 61], [45, 51, 58, 76, 48], [59, 42, 62, 39, 51],
    [60, 66, 71, 73, 61], [55, 53, 42, 26, 69], [56, 46, 50, 38, 47],
    [49, 48, 52, 61, 48], [39, 58, 39, 36, 57], [58, 50, 42, 41, 66],
    [62, 64, 46, 41, 68], [65, 45, 46, 46, 49], [68, 59, 60, 60, 33]
]

V = [
    [52, 44, 40, 57, 45], [37, 53, 30, 33, 59], [50, 40, 39, 38, 45],
    [55, 57, 77, 44, 40], [46, 40, 28, 46, 43], [52, 50, 68, 48, 47],
    [60, 43, 34, 51, 40], [45, 38, 37, 47, 47], [33, 51, 58, 56, 25],
    [67, 47, 64, 45, 54], [45, 51, 58, 76, 48], [59, 42, 62, 39, 51],
    [60, 66, 71, 73, 61], [46, 48, 50, 37, 34], [55, 53, 42, 26, 69]
]

G = [
    [36, 64, 50, 67, 37], [48, 51, 54, 55, 28], [54, 47, 45, 57, 51],
    [46, 57, 50, 45, 54], [30, 47, 47, 41, 52], [49, 44, 57, 61, 54],
    [50, 47, 57, 52, 40], [59, 72, 47, 39, 39], [54, 57, 39, 57, 49],
    [57, 59, 39, 45, 33], [70, 64, 49, 48, 62], [35, 54, 42, 34, 49],
    [42, 48, 34, 54, 51], [70, 39, 44, 41, 41], [50, 62, 43, 47, 49]
]

D_arr = [
    [48, 29, 52, 55, 63], [67, 64, 44, 53, 69], [32, 53, 44, 36, 45],
    [49, 63, 53, 40, 45], [66, 41, 43, 60, 61], [45, 51, 58, 76, 48],
    [59, 42, 62, 39, 51], [60, 66, 71, 73, 61], [46, 48, 50, 37, 34],
    [55, 53, 42, 26, 69], [72, 49, 59, 55, 52], [51, 53, 50, 46, 63],
    [50, 27, 61, 48, 51], [59, 37, 54, 46, 59], [41, 47, 44, 75, 48]
]

VARIANTS = {"А": A, "Б": B, "В": V, "Г": G, "Д": D_arr}


# Допоміжні функції
def flatten(rows):
    return [x for r in rows for x in r]


def compute_h_k(values):
    n = len(values)
    xmin, xmax = min(values), max(values)
    R = xmax - xmin
    h = R / (5.0 * math.log10(n))
    k = math.ceil(R / h) if h > 0 else 1
    return xmin, xmax, R, h, k


def discrete_distribution(values):
    n = len(values)
    c = Counter(values)
    xs = sorted(c.keys())
    rows = []
    for x in xs:
        ni = c[x]
        wi = ni / n
        rows.append({"x_i": x, "n_i": ni, "w_i": wi})
    return rows


def interval_distribution(values, h):
    n = len(values)
    xmin, xmax = min(values), max(values)
    k = math.ceil((xmax - xmin) / h) if h > 0 else 1
    edges = [xmin + i * h for i in range(k + 1)]
    counts = [0] * k
    for v in values:
        idx = min(int((v - xmin) / h), k - 1)
        counts[idx] += 1

    rows = []
    cum = 0
    for i in range(k):
        L = edges[i]
        U = edges[i + 1]
        ni = counts[i]
        wi = ni / n
        dens = wi / h if h > 0 else 0
        cum += ni
        F = cum / n
        rows.append({
            "i": i + 1, "L": L, "U": U, "n_i*": ni, "w_i*": wi,
            "w_i*/h": dens, "N_i*": cum, "F*(U_i)": F
        })
    return rows


def median_discrete(values):
    arr = sorted(values)
    n = len(arr)
    if n % 2 == 1:
        return float(arr[n // 2])
    return (arr[n // 2 - 1] + arr[n // 2]) / 2.0


def mode_discrete(values):
    c = Counter(values)
    maxf = max(c.values())
    raw_modes = sorted([x for x, f in c.items() if f == maxf])

    processed_modes = []
    current_group = [raw_modes[0]]
    for i in range(1, len(raw_modes)):
        if raw_modes[i] == current_group[-1] + 1:
            current_group.append(raw_modes[i])
        else:
            processed_modes.append(sum(current_group) / len(current_group))
            current_group = [raw_modes[i]]
    processed_modes.append(sum(current_group) / len(current_group))
    return processed_modes, raw_modes, maxf


def median_interval(interval_rows, h, n):
    target = n / 2.0
    prev_cum = 0
    for r in interval_rows:
        if r["N_i*"] >= target:
            L = r["L"]
            f = r["n_i*"]
            return L + ((target - prev_cum) / f) * h
        prev_cum = r["N_i*"]
    return interval_rows[-1]["U"]


def mode_interval(interval_rows, h):
    freqs = [r["n_i*"] for r in interval_rows]
    m = max(range(len(freqs)), key=lambda i: freqs[i])
    L = interval_rows[m]["L"]
    f_m = freqs[m]
    f_prev = freqs[m - 1] if m - 1 >= 0 else 0
    f_next = freqs[m + 1] if m + 1 < len(freqs) else 0
    denom = 2 * f_m - f_prev - f_next
    if denom == 0:
        return None
    return L + ((f_m - f_prev) / denom) * h


def stats_from_discrete(discrete_rows):
    xbar = sum(r["x_i"] * r["w_i"] for r in discrete_rows)
    Dv = sum(((r["x_i"] - xbar) ** 2) * r["w_i"] for r in discrete_rows)
    sigma = math.sqrt(Dv)
    Vv = (sigma / xbar) * 100 if xbar != 0 else float("inf")
    return xbar, Dv, sigma, Vv


def stats_from_interval(interval_rows):
    mids = [((r["L"] + r["U"]) / 2.0) for r in interval_rows]
    xbar = sum(mids[i] * interval_rows[i]["w_i*"] for i in range(len(interval_rows)))
    Dv = sum(((mids[i] - xbar) ** 2) * interval_rows[i]["w_i*"] for i in range(len(interval_rows)))
    sigma = math.sqrt(Dv)
    Vv = (sigma / xbar) * 100 if xbar != 0 else float("inf")
    return xbar, Dv, sigma, Vv


def save_polygon(discrete_rows, path, title):
    xs = [r["x_i"] for r in discrete_rows]
    ns = [r["n_i"] for r in discrete_rows]
    plt.figure()
    plt.plot(xs, ns, marker="o")
    plt.title(title)
    plt.xlabel("x_i")
    plt.ylabel("n_i")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_histogram(interval_rows, path, title):
    lefts = [r["L"] for r in interval_rows]
    widths = [r["U"] - r["L"] for r in interval_rows]
    heights = [r["w_i*/h"] for r in interval_rows]
    plt.figure()
    plt.bar(lefts, heights, width=widths, align="edge", edgecolor="black")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("w_i*/h")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def control_chart_one(variant_means, Tnom, sigma, n_sub, path, title):
    se = sigma / math.sqrt(n_sub)
    warn_low, warn_high = Tnom - 2 * se, Tnom + 2 * se
    ctrl_low, ctrl_high = Tnom - 3 * se, Tnom + 3 * se
    labels = list(variant_means.keys())
    means = [variant_means[k] for k in labels]
    x = list(range(1, len(labels) + 1))

    plt.figure()
    plt.plot(x, means, marker="o", label="Середні вибіркові")
    plt.axhline(Tnom, label="Центральна лінія")
    plt.axhline(ctrl_high, linestyle="-.", label="Верхня межа регулювання")
    plt.axhline(ctrl_low, linestyle="-.", label="Нижня межа регулювання")
    plt.axhline(warn_high, linestyle="--", label="Верхня попереджувальна межа")
    plt.axhline(warn_low, linestyle="--", label="Нижня попереджувальна межа")
    plt.title(title)
    plt.xlabel("Номер вибірки")
    plt.ylabel("Опір, кОм")
    plt.xticks(x, labels)
    plt.grid(True)
    plt.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.85,
               borderpad=0.4, labelspacing=0.3, handlelength=1.6, handletextpad=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return warn_low, warn_high, ctrl_low, ctrl_high


def print_discrete(discrete_rows):
    print(f"{'x_i':>5} {'n_i':>5} {'w_i':>10}")
    for r in discrete_rows:
        print(f"{r['x_i']:>5} {r['n_i']:>5} {r['w_i']:>10.6f}")


def print_interval(interval_rows):
    header = (
        f"{'Інтервал':>18} {'L':>8} {'U':>8} {'n_i*':>6} "
        f"{'w_i*':>8} {'w_i*/h':>10} {'N_i*':>6} {'F*(U_i)':>8}"
    )
    print(header)
    for i, r in enumerate(interval_rows):
        is_last = (i == len(interval_rows) - 1)
        interval = f"[{r['L']:.2f}; {r['U']:.2f}]" if is_last else f"[{r['L']:.2f}; {r['U']:.2f})"
        print(
            f"{interval:>18} {r['L']:>8.2f} {r['U']:>8.2f} {r['n_i*']:>6} "
            f"{r['w_i*']:>8.4f} {r['w_i*/h']:>10.4f} {r['N_i*']:>6} {r['F*(U_i)']:>8.4f}"
        )


def main(out_dir="lab1_out_var3", Tnom=50.0, sigma_nom=2.5, subgroup_n=75):
    os.makedirs(out_dir, exist_ok=True)
    variant_means = {}

    for name, rows in VARIANTS.items():
        values = flatten(rows)
        n = len(values)
        xmin, xmax, R, h, k = compute_h_k(values)
        var_series = sorted(values)

        discrete_rows = discrete_distribution(values)
        interval_rows = interval_distribution(values, h)

        Me_d = median_discrete(values)
        processed_modes_d, raw_modes_d, fmax = mode_discrete(values)
        Me_i = median_interval(interval_rows, h, n)
        Mo_i = mode_interval(interval_rows, h)

        poly_path = os.path.join(out_dir, f"polygon_{name}.png")
        hist_path = os.path.join(out_dir, f"hist_{name}.png")

        save_polygon(discrete_rows, poly_path, f"Полігон частот (випадок {name})")
        save_histogram(interval_rows, hist_path, f"Гістограма щільності (випадок {name})")

        xbar_d, D_d, sig_d, V_d = stats_from_discrete(discrete_rows)
        xbar_i, D_i, sig_i, V_i = stats_from_interval(interval_rows)

        overall_mean = sum(values) / n
        variant_means[name] = overall_mean

        print("=" * 90)
        print(f"Випадок {name}: n={n}")
        print(f"x_min={xmin}, x_max={xmax}, R={R}, lg(n)={math.log10(n):.4f}, h={h:.6f}, k={k}")
        print("\nПункт 1: Варіаційний ряд (усі 75 значень):")
        print(" ".join(str(x) for x in var_series))
        print("\nДискретний статистичний ряд (усі рядки):")
        print_discrete(discrete_rows)
        print("\nІнтервальний статистичний ряд (усі інтервали):")
        print_interval(interval_rows)

        print("\nПункт 2: R, Me*, Mo*")
        print(f"R = {R}")
        print(f"Me* (дискретний) = {Me_d:.4f}")

        mode_str = ", ".join(f"{m:g}" for m in processed_modes_d)
        print(f"Mo* (дискретний) = {mode_str} (f_max={fmax})")
        if len(raw_modes_d) > 1:
            raw_mode_str = ", ".join(str(m) for m in raw_modes_d)
            if raw_modes_d != processed_modes_d:
                print(f"  (сирі моди до усереднення суміжних: [{raw_mode_str}])")
            else:
                print(f"  (усі моди: [{raw_mode_str}])")

        print(f"Me* (інтервальний) = {Me_i:.4f}")
        print(f"Mo* (інтервальний) = {Mo_i:.4f}" if Mo_i is not None else "Mo* (інтервальний) = None")

        print("\nПункт 3: графіки збережено ->")
        print(f"  {poly_path}")
        print(f"  {hist_path}")

        print("\nПункт 4: x_bar, D, sigma, V")
        print("  4.1 За ДИСКРЕТНИМ рядом:")
        print(f"    x_bar = {xbar_d:.4f}\n    D = {D_d:.4f}\n    sigma = {sig_d:.4f}\n    V = {V_d:.2f}%")
        print("  4.2 За ІНТЕРВАЛЬНИМ рядом:")
        print(f"    x_bar* = {xbar_i:.4f}\n    D* = {D_i:.4f}\n    sigma* = {sig_i:.4f}\n    V* = {V_i:.2f}%")

    cc_path = os.path.join(out_dir, "control_chart_all.png")
    # Для карти середніх арифметичних n_sub = 75
    warn_low, warn_high, ctrl_low, ctrl_high = control_chart_one(
        variant_means, Tnom=Tnom, sigma=sigma_nom, n_sub=subgroup_n,
        path=cc_path, title="Контрольна карта середніх арифметичних"
    )

    print("\n" + "=" * 90)
    print("Пункт 6: ОДНА контрольна карта середніх арифметичних (А, Б, В, Г, Д)")
    print(f"Попереджувальні межі: [{warn_low:.4f}; {warn_high:.4f}]")
    print(f"Межі регулювання: [{ctrl_low:.4f}; {ctrl_high:.4f}]")
    print("Середні для вибірок (А, Б, В, Г, Д): " + ", ".join(f"{k}={v:.4f}" for k, v in variant_means.items()))
    print(f"Графік збережено -> {cc_path}")
    print("\nГотово! Усі картинки лежать у папці:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
