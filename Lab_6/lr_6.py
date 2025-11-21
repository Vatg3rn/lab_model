import random
import math
import numpy as np


def simulate_detection(p0, p1, p_interfere, n_cycles):
    for _ in range(n_cycles):
        if random.random() < p_interfere:
            prob_detect = p1
        else:
            prob_detect = p0
        if random.random() < prob_detect:
            return True
    return False


def monte_carlo_probability(p0, p1, p_interfere, n_cycles, num_simulations=100000):
    detected_count = 0
    for _ in range(num_simulations):
        if simulate_detection(p0, p1, p_interfere, n_cycles):
            detected_count += 1

    prob_estimate = detected_count / num_simulations
    return prob_estimate


def confidence_interval(success_count, num_simulations, confidence=0.95):
    p_hat = success_count / num_simulations
    z = 1.96

    std_error = math.sqrt(p_hat * (1 - p_hat) / num_simulations)
    margin = z * std_error

    lower = p_hat - margin
    upper = p_hat + margin

    return lower, upper


def analytical_probability(p0, p1, p_interfere, n_cycles):
    prob_not_detect_one_cycle = p_interfere * (1 - p1) + (1 - p_interfere) * (1 - p0)
    prob_not_detect_n_cycles = prob_not_detect_one_cycle ** n_cycles
    prob_detect_at_least_once = 1 - prob_not_detect_n_cycles
    return prob_detect_at_least_once


p0 = 0.8
p1 = 0.3
p_interfere = 0.4
n_cycles = 5
num_simulations = 100000

mc_prob = monte_carlo_probability(p0, p1, p_interfere, n_cycles, num_simulations)

success_count = int(mc_prob * num_simulations)
ci_lower, ci_upper = confidence_interval(success_count, num_simulations)

analytical_prob = analytical_probability(p0, p1, p_interfere, n_cycles)

print(f"Метод Монте-Карло:")
print(f"Оценка вероятности: {mc_prob:.6f}")
print(f"95% доверительный интервал: [{ci_lower:.6f}, {ci_upper:.6f}]")
print()
print(f"Аналитическое решение: {analytical_prob:.6f}")
print()

print("Разница (МК - аналит.):", abs(mc_prob - analytical_prob))