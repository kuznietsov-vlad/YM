import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import pandas as pd

plt.style.use('seaborn-v0_8')  # стиль графіків

# =========================
# Вхідні дані
# =========================
bins = [(8, 9), (9, 10), (10, 11), (11, 12),
        (12, 13), (13, 14), (14, 15), (15, 16)]

ni = np.array([10, 8, 7, 10, 22, 18, 14, 11])
n = ni.sum()
h = 1

density = ni / (n * h)
left_edges = [b[0] for b in bins]

# =========================
# Гістограма
# =========================
plt.figure(figsize=(8, 5))

plt.bar(left_edges, density, width=h, align='edge',
        edgecolor='black', linewidth=1.2,
        color='#4C72B0')

plt.xlabel("Інтервали", fontsize=11)
plt.ylabel("Щільність", fontsize=11)
plt.title("Гістограма щільності розподілу", fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# =========================
# ПІРСОН
# =========================
midpoints = np.array([(a + b) / 2 for a, b in bins])

x_bar = np.sum(ni * midpoints) / n
D = np.sum(ni * (midpoints - x_bar) ** 2) / n
sigma = np.sqrt(D)

print("\n--- Статистичні характеристики ---")
print(f"Середнє: {x_bar:.4f}")
print(f"Сигма: {sigma:.4f}")

xi = np.array([a for a, _ in bins])
xip1 = np.array([b for _, b in bins])

z_i = (xi - x_bar) / sigma
z_ip1 = (xip1 - x_bar) / sigma

z_i[0] = -np.inf
z_ip1[-1] = np.inf

def laplas(z):
    val = 0.5 * erf(z / np.sqrt(2))
    val[np.isneginf(z)] = -0.5
    val[np.isposinf(z)] = 0.5
    return val

Phi_i = laplas(z_i)
Phi_ip1 = laplas(z_ip1)
P = Phi_ip1 - Phi_i

df = pd.DataFrame({
    "xi": xi,
    "xi+1": xip1,
    "zi": z_i,
    "zi+1": z_ip1,
    "Φ(zi)": Phi_i,
    "Φ(zi+1)": Phi_ip1,
    "Pi": P
}).round(4)

print("\n--- Таблиця інтервалів ---")
print(df.to_string(index=False))

print(f"\nΣ Pi = {P.sum():.6f}")

# =========================
# χ² Пірсона
# =========================
n_theor = n * P
diff = ni - n_theor
chi_parts = (diff ** 2) / n_theor

df2 = pd.DataFrame({
    "i": np.arange(1, len(ni) + 1),
    "ni": ni,
    "ni'": n_theor,
    "ni - ni'": diff,
    "(ni - ni')^2 / ni'": chi_parts
}).round(4)

df2.loc["Σ"] = ["", n, n, "", chi_parts.sum()]

print("\n--- Критерій Пірсона ---")
print(df2.to_string(index=False))

# =========================
# КОЛМОГОРОВ
# =========================
# межі інтервалів
right_edges = np.array([b for _, b in bins])
xi = np.concatenate(([-np.inf], right_edges))
xi1 = np.concatenate((right_edges, [np.inf]))

# частоти
ni_full = np.concatenate(([0], ni))
nx = np.cumsum(ni_full)

# емпірична функція
F_emp = nx / n

# z для F(x_i)
z = (xi - x_bar) / sigma

# теоретична функція
F_theor = np.where(
    np.isneginf(z), 0,
    np.where(np.isposinf(z), 1, 0.5 + laplas(z))
)

# різниця
abs_diff = np.abs(F_emp - F_theor)
F_theor[-1] = 1
abs_diff[-1] = 0

# статистика Колмогорова
D_kolm = abs_diff.max()
lambda_obs = D_kolm * np.sqrt(n)

# таблиця
table = pd.DataFrame({
    "i": np.arange(1, len(xi) + 1),
    "xi": xi,
    "xi+1": xi1,
    "ni": ni_full,
    "nx": nx,
    "F*(x)": F_emp,
    "F(x)": F_theor,
    "|F*-F|": abs_diff
})

table["xi"] = table["xi"].replace(-np.inf, "-inf")
table["xi+1"] = table["xi+1"].replace(np.inf, "+inf")

print("\n--- Колмогоров ---")
print(table.to_string(index=False))

lambda_crit = 1.36  # критичне значення

print("\n--- ВИСНОВОК ---")
print("n =", n)
print("D =", round(D_kolm, 6))
print("Спостережуване значення статистики Колмогорова =", round(lambda_obs, 6))
print("Критичне значення статистики Колмогорова =", lambda_crit)

if lambda_obs < lambda_crit:
    print("ВИСНОВОК: H0 НЕ відхиляємо")
else:
    print("ВИСНОВОК: H0 ВІДХИЛЯЄМО")

# =========================
# Графік + нормальний розподіл
# =========================
plt.figure(figsize=(8, 5))

plt.bar(left_edges, density, width=h, align='edge',
        alpha=0.6, edgecolor='black',
        color='#4C72B0', label="Гістограма")

x_vals = np.linspace(bins[0][0], bins[-1][1], 400)

pdf_vals = (1 / (sigma * np.sqrt(2 * np.pi))) * \
           np.exp(-(x_vals - x_bar) ** 2 / (2 * sigma ** 2))

plt.plot(x_vals, pdf_vals, color='#DD8452',
         linewidth=2.5, label="Нормальний розподіл")

plt.xlabel("x")
plt.ylabel("Щільність")
plt.title("Порівняння з нормальним розподілом")

plt.legend()
plt.grid(alpha=0.4)

plt.tight_layout()
plt.show()

# =========================
# 3-й графік: функції розподілу
# =========================
plt.figure(figsize=(8, 5))

# емпірична функція (ступінчаста)
plt.step(xi, F_emp, where='post',
         linewidth=2, color='#4C72B0',
         label="Емпірична F*(x)")

# теоретична функція
plt.plot(xi, F_theor,
         linewidth=2.5, color='#DD8452',
         label="Теоретична F(x)")

plt.xlabel("x")
plt.ylabel("F(x)")
plt.title("Порівняння емпіричної та теоретичної функцій розподілу")

plt.legend()
plt.grid(alpha=0.4)

plt.tight_layout()
plt.show()

