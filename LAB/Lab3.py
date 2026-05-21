import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Дані для 8 варіанту: звуковий канал (X) та відеоканал (Y)
X = np.array([170, 180, 200, 230, 240, 250, 280, 300, 310, 320, 330, 350, 380, 400, 410, 420, 430, 440, 450, 460])
Y = np.array([240, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 65, 60, 55, 50, 45])
n = len(X)

#1. Точкові незміщені оцінки параметрів
X_mean = np.mean(X)
Y_mean = np.mean(Y)

# Сума квадратів відхилень X
ss_x = np.sum((X - X_mean)**2)

# Коефіцієнти b1 та b0
b1 = np.sum((X - X_mean) * (Y - Y_mean)) / ss_x
b0 = Y_mean - b1 * X_mean

print(f"1. Точкові оцінки: b0 = {b0:.4f}, b1 = {b1:.4f}")
print(f"Рівняння регресії: y = {b0:.4f} + ({b1:.4f}) * x")

# Графік 1: Кореляційне поле та лінія регресії
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Кореляційне поле')
plt.plot(X, b0 + b1 * X, color='red', label='Лінія регресії')
plt.title("Кореляційне поле та лінія регресії")
plt.xlabel("X (Звуковий канал)")
plt.ylabel("Y (Відеоканал)")
plt.grid(True)
plt.legend()
plt.show()

#2. Довірчі інтервали для параметрів (gamma = 0.99)
gamma = 0.99
alpha = round(1 - gamma, 2)

# Критичне значення t-розподілу Стьюдента
t_crit = stats.t.ppf(1 - alpha/2, df=n - 2)

# Прогнозовані значення Y та залишкова дисперсія
Y_pred = b0 + b1 * X
s_squared = np.sum((Y - Y_pred)**2) / (n - 2)

# Стандартні похибки для параметрів
std_b1 = np.sqrt(s_squared / ss_x)
std_b0 = np.sqrt(s_squared * (1/n + (X_mean**2) / ss_x))

print(f"\n2. Довірчі інтервали (gamma = {gamma}):")
print(f"b0: [{b0 - t_crit * std_b0:.4f}; {b0 + t_crit * std_b0:.4f}]")
print(f"b1: [{b1 - t_crit * std_b1:.4f}; {b1 + t_crit * std_b1:.4f}]")

#3. Перевірка значущості параметра b0 (alpha = 0.01)
t_obs_b0 = b0 / std_b0

print(f"\n3. Перевірка значущості b0:")
print(f"t_спост = {t_obs_b0:.4f}")
print(f"t_табл = {t_crit:.4f}")

if abs(t_obs_b0) > t_crit:
    print("Висновок: Параметр b0 значущий")
else:
    print("Висновок: Параметр b0 незначущий")

#4. Довірчий інтервал для функції регресії
# Сортуємо X для плавного відображення меж на графіку
idx = np.argsort(X)
X_s = X[idx]
Y_p = Y_pred[idx]

# Стандартна похибка лінії регресії для кожного x
SE_line = np.sqrt(s_squared * (1/n + (X_s - X_mean)**2 / ss_x))

# Межі довірчого інтервалу
low_b = Y_p - t_crit * SE_line
high_b = Y_p + t_crit * SE_line

print(f"\n4. Довірчий інтервал для функції регресії:")
for i in range(len(X_s)):
    print(f"Для y({X_s[i]}): [{low_b[i]:.4f}; {high_b[i]:.4f}]")

# Графік 2: Довірчий інтервал функції регресії
plt.figure(figsize=(10, 6))
plt.scatter(X_s, Y[idx], label='Кореляційне поле')
plt.plot(X_s, Y_p, color='red', label='Лінія регресії')
plt.fill_between(X_s, low_b, high_b, color='purple', alpha=0.2, label=f'Довірчий інтервал (γ={gamma})')
plt.title("Довірчий інтервал для функції регресії")
plt.xlabel("X (Звуковий канал)")
plt.ylabel("Y (Відеоканал)")
plt.grid(True)
plt.legend()
plt.show()

#5. Парний коефіцієнт кореляції
r_xy = np.corrcoef(X, Y)[0, 1]
abs_r = abs(r_xy)

if abs_r < 0.3:
    strength = "дуже слабкий"
elif abs_r < 0.5:
    strength = "слабкий"
elif abs_r < 0.7:
    strength = "помірний"
elif abs_r < 0.9:
    strength = "сильний"
else:
    strength = "дуже сильний"

print(f"\n5. Коефіцієнт кореляції r_xy = {r_xy:.4f}")
print(f"Сила зв'язку: {strength}")
