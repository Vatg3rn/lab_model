import math

# Инициализация переменных
randoml = [0.0] * 6000  # Индексация с 1 до 6000 — для удобства делаем список длиной 6001
count = [0] * 50      # Индексы 1..16 — делаем длиной 17

# Начальные значения
randoml[0] = 0.159819
randoml[1] = 0.901967
randoml[2] = 0.251897
randoml[3] = 0.412358
randoml[4] = 0.819203
randoml[5] = 0.398273
randoml[6] = 0.264839
randoml[7] = 0.583729
randoml[8] = 0.482712
randoml[9] = 0.385722

s = 0.0
sqrs = 0.0
hi = 0.0
xlow = 0
dmax = 0.0


def kolmogorov_test(data):
    n = len(data)
    if n == 0:
        raise ValueError("Выборка пуста")

    sorted_data = sorted(data)

    def cdf_theoretical(x):
        if x < 0:
            return 0.0
        elif x > 1:
            return 1.0
        else:
            return x

    D_plus = 0.0
    D_minus = 0.0

    for i, x in enumerate(sorted_data, start=1):
        F_theor = cdf_theoretical(x)
        F_emp_plus = i / n
        F_emp_minus = (i - 1) / n

        D_plus = max(D_plus, F_emp_plus - F_theor)
        D_minus = max(D_minus, F_theor - F_emp_minus)

    D = max(D_plus, D_minus)
    lambda_k = D * math.sqrt(n)

    return lambda_k

def getR(x, n, p):
    R = 0
    # определяем 0-й элемент последовательности y
    y1 = x[0] >= p  # эквивалентно: если x[0] < p, то False, иначе True

    for i in range(1, n):
        # определяем i-й элемент последовательности y
        y2 = x[i] >= p  # аналогично

        if y1 != y2:
            R += 1  # увеличиваем число R

        y1 = y2  # обновляем предыдущее значение
    mat_ojid = 2*n*p*(1-p)+p**2+(1-p)**2
    sko = math.sqrt((4*n*p*(1-p))*(1-3*p*(1-p)) - 2*p*(1-p)*(3-10*p*(1-p)))
    Rl = mat_ojid - 1.96*sko
    Rh = mat_ojid + 1.96*sko
    return R, Rl, Rh

for i in range(10, 6000):
    new_value = 0.0
    for j in range(i-1, i-10, -1):
        new_value += randoml[j]
    randoml[i] = new_value % 1

for i in range(6000):
    s += randoml[i]
    sqrs += randoml[i] ** 2
    z = 0
    while True:
        z += 1
        lower_bound = (z - 1) / 16.0
        upper_bound = z / 16.0
        if lower_bound <= randoml[i] <= upper_bound:
            count[int(z)] += 1
            break

s /= 6000.0

print(f"Распределение чисел по интервалам[{1/16}]:")
for i in range(1, 17):
    xlow += count[i]
    norm_freq = (count[i] / 6000.0) * 16
    cum_freq = xlow / 6000.0
    print(f"{i:2d}-ый интервал: {count[i]}, норм. частота: {norm_freq:6.5f}, меньше или равно: {cum_freq:5.6f}")
    hi += (count[i] - 6000/16) ** 2

print(f"Выборочная средняя: {s:5.6f}")
print("Математическое ожидание: 0.5")
variance_unbiased = (sqrs / 6000.0) - (s ** 2)
print(f"Несмещенная оценка дисперсии: {variance_unbiased:5.6f}")
print(f"Требуемая дисперсия: {1/12:5.6f}")
print(f"Коэффициент ХИ-квадрат равен: {(hi / 375.0):5.6f}, критический для уровня значимости 0.05: 24.996")
R, Rl, Rh = getR(randoml, 6000, 0.4)
print(f"Интервал теста серии длин для уровня значимости 0.05: {int(Rl)} < {R} <{int(Rh)}")
lambda_kolmogorov = kolmogorov_test(randoml)
print(f"Коэффициент лямбда равен: {lambda_kolmogorov:5.6f}, критический для уровня значимости 0.05: 1.358")