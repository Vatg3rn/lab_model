import math
import random
import matplotlib.pyplot as plt
import numpy as np

N = 2000
K = 15


def generate_sample(n):
    U = [0.0] * n
    for i in range(n):
        U[i] = random.random()
    X = [0.0] * n
    for i in range(n):
        u = U[i]
        if u < 0.4:
            X[i] = (20 * u) ** (1 / 3)
        else:
            X[i] = -math.log((1 - u) / 2) / 0.602
    return X


sample = generate_sample(N)

count = [0] * (K + 1)
s = 0.0
sqrs = 0.0
hi = 0.0
xlow = 0

x_min = min(sample)
x_max = max(sample)
interval_width = (x_max - x_min) / K


for i in range(N):
    x = sample[i]
    s += x
    sqrs += x * x
    if x >= x_max:
        bin_index = K
    else:
        bin_index = int((x - x_min) / interval_width) + 1
        if bin_index < 1:
            bin_index = 1
        if bin_index > K:
            bin_index = K
    count[bin_index] += 1

mean_est = s / N


def F_theoretical(x):
    if x < 0:
        return 0.0
    elif x < 2:
        return 0.05 * x ** 3
    else:
        return 1 - 2 * math.exp(-0.602 * x)


theoretical_probs = []
for i in range(K):
    left = x_min + i * interval_width
    right = x_min + (i+1) * interval_width
    p = F_theoretical(right) - F_theoretical(left)
    theoretical_probs.append(p)

print(f"Распределение чисел по {K} интервалам в диапазоне [{x_min:.4f}, {x_max:.4f}]:")
for i in range(K):
    xlow += count[i+1]
    observed_freq = count[i+1]
    expected_freq = N * theoretical_probs[i]
    norm_freq = observed_freq / (N * interval_width)  # оценка плотности
    cum_freq = xlow / N
    print(f"{i+1:2d}-ый интервал: {observed_freq:4d}, экспериментальная частота: {norm_freq:8.5f}, теоретическая частота: {cum_freq:7.6f}")
    if expected_freq > 0:
        hi += (observed_freq - expected_freq) ** 2 / expected_freq

variance_unbiased = (sqrs / N) - (mean_est ** 2)

print(f"\nВыборочная средняя: {mean_est:8.6f}")
print(f"Несмещённая оценка дисперсии: {variance_unbiased:8.6f}")
print(f"\nКоэффициент ХИ-квадрат равен: {hi:8.6f}, критическое значение = 23.685")

def kolmogorov_test(data):
    n = len(data)
    if n == 0:
        raise ValueError("Выборка пуста")
    sorted_data = sorted(data)
    D_plus = 0.0
    D_minus = 0.0
    for i, x in enumerate(sorted_data, start=1):
        F_theor = F_theoretical(x)
        F_emp_plus = i / n
        F_emp_minus = (i - 1) / n
        D_plus = max(D_plus, F_emp_plus - F_theor)
        D_minus = max(D_minus, F_theor - F_emp_minus)
    D = max(D_plus, D_minus)
    lambda_k = D * math.sqrt(n)
    return lambda_k


lambda_kolmogorov = kolmogorov_test(sample)
print(f"\nКоэффициент λ Колмогорова: {lambda_kolmogorov:8.6f}, критическое значение при a = 0.05 = 1.358")
