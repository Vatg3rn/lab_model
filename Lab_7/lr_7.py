import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math


max_j = 1000
a = 6 / (math.pi ** 2)
j_values = np.arange(1, max_j + 1)
probabilities = a / (j_values ** 2)
probabilities /= probabilities.sum()


def generate_walk_distance(n_steps=8):
    step_lengths = np.random.choice(j_values, size=n_steps, p=probabilities)
    directions = np.random.choice([-1, 1], size=n_steps)
    final_position = np.sum(step_lengths * directions)
    return abs(final_position)


num_simulations = 20000
distances = np.array([generate_walk_distance() for _ in range(num_simulations)])


exp_scale = np.mean(distances)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(distances, bins=50, density=True, alpha=0.7, edgecolor='black', label='Данные')
x = np.linspace(0, distances.max(), 500)
plt.plot(x, stats.expon.pdf(x, scale=exp_scale), 'r-', lw=2, label=f'Экспоненциальная\n(scale={exp_scale:.2f})')
plt.xlabel('Удаление |x|')
plt.ylabel('Плотность')
plt.title('Гистограмма и экспоненциальная PDF')
plt.legend()
plt.grid(True, alpha=0.3)


plt.subplot(1, 2, 2)
sorted_data = np.sort(distances)
y_empirical = np.arange(1, len(sorted_data)+1) / len(sorted_data)
plt.plot(sorted_data, y_empirical, 'b-', drawstyle='steps-post', label='Эмпирическая ФР')
plt.plot(sorted_data, stats.expon.cdf(sorted_data, scale=exp_scale), 'r--', lw=2, label='Экспоненциальная ФР')
plt.xlabel('Удаление |x|')
plt.ylabel('F(x)')
plt.title('Функции распределения')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
stats.probplot(distances, dist=stats.expon, sparams=(0, exp_scale), plot=plt)
plt.xlabel('Теоретические квантили')
plt.ylabel('Эмпирические квантили')
plt.title('Q-Q Plot: Данные vs Экспоненциальное распределение')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Среднее удаление: {np.mean(distances):.2f}")
print(f"Оценка параметра экспоненциального распределения (scale = 1/lambda): {exp_scale:.2f}")