import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math


a = 6 / (math.pi ** 2)
max_j = 100
j_values = np.arange(1, max_j + 1)
probs = a / (j_values ** 2)
probs /= probs.sum()

def generate_step():
    j = np.random.choice(j_values, p=probs)
    direction = np.random.choice([-1, 1])
    return j * direction


def random_walk_1d(n_steps):
    position = 0
    for _ in range(n_steps):
        step = generate_step()
        position += step
    return abs(position)


def run_simulation(n_simulations, n_steps):
    distances = []
    for _ in range(n_simulations):
        dist = random_walk_1d(n_steps)
        distances.append(dist)
    return np.array(distances)


def estimate_sample_size_for_mean(data, epsilon, gamma):
    n = len(data)
    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1)
    if sample_var == 0:
        sample_var = 1e-6
    z_gamma = norm.ppf((1 + gamma) / 2)
    n_required = int(math.ceil((z_gamma ** 2) * sample_var / (epsilon ** 2)))
    return max(n_required, n), sample_mean


def estimate_sample_size_for_variance(data, epsilon, gamma):
    n = len(data)
    s2 = np.var(data, ddof=1)
    fourth_moment = np.mean((data - np.mean(data))**4)
    var_s2 = (fourth_moment - (s2**2)) / n
    if var_s2 <= 0:
        var_s2 = 1e-6
    z_gamma = norm.ppf((1 + gamma) / 2)
    n_required = int(math.ceil((z_gamma ** 2) * var_s2 / (epsilon ** 2)))
    return max(n_required, n), s2


def plot_histogram(data, title):
    plt.hist(data, bins=50, density=True, alpha=0.6, edgecolor='black', label='Данные')
    plt.title(title)
    plt.xlabel('Удаление')
    plt.ylabel('Плотность')
    plt.grid(True, alpha=0.3)
    plt.show()


n_steps = 8
epsilon_mean = 0.5  # Точность для среднего
gamma = 0.95        # Достоверность
epsilon_var = 0.3   # Точность для дисперсии

print("Запуск симуляции...")
pilot_size = 1000
pilot_data = run_simulation(pilot_size, n_steps)

print(f"Пробное среднее значение: {np.mean(pilot_data):.2f}")
print(f"Пробное СКО: {np.std(pilot_data):.2f}")
print(f"Пробная диспесия: {np.var(pilot_data):.2f}")


plot_histogram(pilot_data, "Гистограмма")


print("\nЧасть 1: среднее значение")
n_needed_mean, est_mean = estimate_sample_size_for_mean(pilot_data, epsilon_mean, gamma)
print(f"Необходимый размер выборки: {n_needed_mean}")
if n_needed_mean > pilot_size:
    additional = run_simulation(n_needed_mean - pilot_size, n_steps)
    full_data = np.concatenate([pilot_data, additional])
else:
    full_data = pilot_data
final_mean = np.mean(full_data)
print(f"Оцененное среднее значение: {final_mean:.2f}")


print("\nЧасть 2: дисперсия")
n_needed_var, est_var = estimate_sample_size_for_variance(full_data, epsilon_var, gamma)
print(f"Необходимый размер выборки: {n_needed_var}")
if n_needed_var > len(full_data):
    additional = run_simulation(n_needed_var - len(full_data), n_steps)
    final_data = np.concatenate([full_data, additional])
else:
    final_data = full_data
final_var = np.var(final_data, ddof=1)
print(f"Оцененная диспресия: {final_var:.2f}")
