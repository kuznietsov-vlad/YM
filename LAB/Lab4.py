import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

data = {
    "Y":  [8.99, 12.28, 8.00, 7.27, 7.47, 10.86, 5.23, 12.16, 9.19, 10.12, 6.86, 11.02, 7.77, 10.62, 7.40],
    "X1": [37, 33, 25, 29, 53, 41, 26, 32, 59, 48, 51, 43, 29, 37, 49],
    "X2": [8, 24, 18, 4, 13, 9, 12, 23, 11, 3, 8, 22, 9, 12, 5],
    "X3": [0.06, 0.12, 0.02, 0.07, 0.14, 0.08, 0.13, 0.10, 0.13, 0.09, 0.12, 0.15, 0.02, 0.08, 0.14],
    "X4": [4.3, 5.0, 2.9, 3.5, 2.7, 4.9, 3.4, 4.8, 3.9, 4.8, 2.9, 3.7, 3.5, 5.0, 4.1]
}

gama = 0.99

Y = np.array(data["Y"])
X = np.column_stack((
    np.ones(len(Y)),
    data["X1"],
    data["X2"],
    data["X3"],
    data["X4"]
))

n, m = X.shape

# Завдання 1
XtX_inv = np.linalg.inv(X.T @ X)
B = XtX_inv @ X.T @ Y

print("Завдання 1")
print("Коефіцієнти B:")
for i, b in enumerate(B):
    print(f"B{i} = {b:.4f}")

reg_eq = f"y = {B[0]:.4f}"
for i in range(1, len(B)):
    reg_eq += f" + ({B[i]:.4f})*x{i}"

print("\nРівняння множинної регресії:")
print(reg_eq)

# Завдання 2
Y_hat = X @ B
e = Y - Y_hat

s2 = np.sum(e ** 2) / (n - m)
s = np.sqrt(s2)

t_crit = t.ppf((1 + gama) / 2, n - m)

print("\nЗавдання 2")
print("Довірчі інтервали для значень функції регресії:")

margin_list = []

for i in range(n):
    x0 = X[i]
    margin = t_crit * np.sqrt(s2 * (x0 @ XtX_inv @ x0.T))
    margin_list.append(margin)

    low = Y_hat[i] - margin
    high = Y_hat[i] + margin

    vector_values = tuple(round(float(val), 2) for val in x0[1:])
    print(f"Для вектора значень {vector_values} інтервал [{low:.2f}; {high:.2f}]")

# Завдання 3
Y_bar = np.mean(Y)

numerator = B.T @ X.T @ Y - n * Y_bar ** 2
denominator = Y.T @ Y - n * Y_bar ** 2

R2 = numerator / denominator
R = np.sqrt(R2)

print("\nЗавдання 3")
print(f"Коефіцієнт детермінації R2 = {R2:.4f}")
print(f"Коефіцієнт множинної кореляції R = {R:.4f}")

# Графік
plt.figure(figsize=(10, 6))

indices = np.arange(n)

lower_bound = Y_hat - np.array(margin_list)
upper_bound = Y_hat + np.array(margin_list)

plt.fill_between(
    indices,
    lower_bound,
    upper_bound,
    alpha=0.5,
    label="Довірчий інтервал (99%)"
)

plt.plot(indices, Y_hat, "bo-", label="Множинна регресія (Ŷ)")
plt.plot(indices, Y, "ro", label="Фактичні значення (Y)")

plt.xlabel("Індекс спостереження")
plt.ylabel("Значення Y")
plt.title(
    f"Коефіцієнт детермінації R2 = {R2:.4f}\n"
    f"Коефіцієнт множинної кореляції R = {R:.4f}"
)

plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()