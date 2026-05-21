import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

GAMMA = 0.99

# Завдання 1
x1 = np.array([2.18, 2.27, 2.35, 1.72, 1.72, 2.04, 2.07, 1.92, 1.84, 1.72, 1.95, 2.01, 1.64, 1.38, 1.50])
y1 = np.array([7.22, 6.93, 6.22, 9.56, 9.34, 6.58, 6.95, 7.73, 10.03, 9.26, 8.03, 7.25, 9.05, 10.68, 10.32])

# Завдання 2
y2 = np.array([468, 528, 509, 511, 534, 550, 519, 549, 531, 496, 500, 464, 475, 500, 521])
x2 = np.array([1200, 1450, 1600, 1710, 1850, 1910, 2070, 2210, 2340, 2600, 2700, 2900, 2850, 2700, 2500])
z2 = np.array([600, 620, 580, 520, 430, 390, 340, 410, 550, 330, 410, 480, 560, 530, 570])

# ДОПОМІЖНІ ФУНКЦІЇ
def cheddock_scale(r):
    val = abs(r)
    if val < 0.3: return "Слабкий"
    elif val < 0.5: return "Помірний"
    elif val < 0.7: return "Помітний"
    elif val < 0.9: return "Сильний"
    else: return "Дуже сильний"

# ЗАВДАННЯ 1: Нелінійна (гіперболічна) регресія
print("=== ЗАВДАННЯ 1: Гіперболічна регресія ===")

# Заміна змінної для зведення до лінійної: X_inv = 1 / X
x1_inv = 1 / x1
n1 = len(x1)

# Матричний метод: B = (X^T * X)^-1 * X^T * Y
X_mat1 = np.vstack((np.ones(n1), x1_inv)).T
B1 = np.linalg.inv(X_mat1.T @ X_mat1) @ X_mat1.T @ y1
b0, b1 = B1

print(f"Рівняння регресії: y = {b0:.4f} + {b1:.4f}/x")
print(f"b0 = {b0:.4f}, b1 = {b1:.4f}")

# Розрахунок R та R^2
y1_pred = b0 + b1 * x1_inv
SST1 = np.sum((y1 - np.mean(y1))**2)
SSR1 = np.sum((y1_pred - np.mean(y1))**2)
SSE1 = np.sum((y1 - y1_pred)**2)

R1_sq = SSR1 / SST1
R1 = np.sqrt(R1_sq)

print(f"Коефіцієнт кореляції (R): {R1:.4f} - {cheddock_scale(R1)} зв'язок")
print(f"Коефіцієнт детермінації (R^2): {R1_sq:.4f}")

# Довірчі інтервали для функції (Завдання 1)
df1 = n1 - 2
t_val1 = t.ppf((1 + GAMMA) / 2, df1)
s_resid1 = np.sqrt(SSE1 / df1)

x_mean_inv = np.mean(x1_inv)
Sxx1 = np.sum((x1_inv - x_mean_inv)**2)

print("\nДовірчі інтервали для точок (Завдання 1):")
for i in range(n1):
    delta = t_val1 * s_resid1 * np.sqrt(1/n1 + (x1_inv[i] - x_mean_inv)**2 / Sxx1)
    print(f"X = {x1[i]:.2f} -> ({y1_pred[i] - delta:.4f}, {y1_pred[i] + delta:.4f})")


# ЗАВДАННЯ 2: Множинна нелінійна регресія
print("\n=== ЗАВДАННЯ 2: Множинна регресія ===")

n2 = len(x2)

# Формування матриці X: [1, x, x^2, z, z^2]
X_mat2 = np.vstack((np.ones(n2), x2, x2**2, z2, z2**2)).T
B2 = np.linalg.inv(X_mat2.T @ X_mat2) @ X_mat2.T @ y2

print(f"Рівняння: y = {B2[0]:.4f} + ({B2[1]:.4f})x + ({B2[2]:.6f})x^2 + ({B2[3]:.4f})z + ({B2[4]:.6f})z^2")

for i, b in enumerate(B2):
    print(f"b{i} = {b:.6f}")

# Розрахунок R та R^2
y2_pred = X_mat2 @ B2
SST2 = np.sum((y2 - np.mean(y2))**2)
SSR2 = np.sum((y2_pred - np.mean(y2))**2)
SSE2 = np.sum((y2 - y2_pred)**2)

R2_sq = SSR2 / SST2
R2 = np.sqrt(R2_sq)

print(f"Множинний коефіцієнт кореляції (R): {R2:.4f} - {cheddock_scale(R2)} зв'язок")
print(f"Коефіцієнт детермінації (R^2): {R2_sq:.4f}")

# Довірчі інтервали (Завдання 2)
df2 = n2 - 5
t_val2 = t.ppf((1 + GAMMA) / 2, df2)
sigma2 = SSE2 / df2
Xtx_inv2 = np.linalg.inv(X_mat2.T @ X_mat2)

print("\nДовірчі інтервали для функції (Завдання 2):")
for i in range(n2):
    x_vec = X_mat2[i]
    delta = t_val2 * np.sqrt(sigma2 * (x_vec.T @ Xtx_inv2 @ x_vec))
    print(f"X={x2[i]}, Z={z2[i]} -> ({y2_pred[i] - delta:.4f}, {y2_pred[i] + delta:.4f})")


# 1. Графік для Завдання 1
x_plot = np.linspace(min(x1), max(x1), 100)
y_plot = b0 + b1 * (1 / x_plot)

plt.figure(figsize=(10, 6))
plt.scatter(x1, y1, color='red', label='Вихідні дані')
plt.plot(x_plot, y_plot, color='blue', label=f'Гіперболічна регресія: y = {b0:.3f} + {b1:.3f}/x')
plt.title('Завдання 1: Кореляційне поле та графік нелінійної регресії')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# 2. Графік для Завдання 1 з довірчими інтервалами
delta_plot = t_val1 * s_resid1 * np.sqrt(1/n1 + ((1/x_plot) - x_mean_inv)**2 / Sxx1)
y_lower = y_plot - delta_plot
y_upper = y_plot + delta_plot

plt.figure(figsize=(10, 6))
plt.scatter(x1, y1, color='red', label='Вихідні дані')
plt.plot(x_plot, y_plot, color='blue', label='Регресійна крива')
plt.fill_between(x_plot, y_lower, y_upper, color='green', alpha=0.2, label='Довірчий інтервал 99%')
plt.title('Завдання 1: Гіперболічна регресія з довірчим інтервалом')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# 3. 3D Графік для Завдання 2
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Сітка для побудови поверхні
X_grid = np.linspace(min(x2), max(x2), 30)
Z_grid = np.linspace(min(z2), max(z2), 30)

X_mesh, Z_mesh = np.meshgrid(X_grid, Z_grid)

Y_mesh = B2[0] + B2[1]*X_mesh + B2[2]*(X_mesh**2) + B2[3]*Z_mesh + B2[4]*(Z_mesh**2)

ax.scatter(x2, z2, y2, color='red', s=50, label='Вихідні точки', alpha=1)
ax.plot_surface(X_mesh, Z_mesh, Y_mesh, cmap='viridis', alpha=0.6)

ax.set_title('Завдання 2: Поверхня множинної нелінійної регресії')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
ax.view_init(elev=20, azim=-45)

plt.legend()
plt.show()