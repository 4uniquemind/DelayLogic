import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------
# 1. Модель "Круговой Карты" (Аналогия)
# -------------------------------------------
# theta_next = (theta_current + Omega + (K / (2 * pi)) * sin(2 * pi * theta_current)) mod 1
# Omega - аналог отношения собственной частоты к частоте драйвера (f0/fm в статье)
# K - сила связи / нелинейности (аналог глубины модуляции epsilon в статье)
# theta - фаза системы (от 0 до 1)

def circle_map_step(theta_current, Omega, K):
    """Один шаг итерации круговой карты."""
    theta_next = theta_current + Omega + (K / (2 * np.pi)) * np.sin(2 * np.pi * theta_current)
    return theta_next % 1 # Возвращаем остаток от деления на 1, чтобы фаза была [0, 1)

# -------------------------------------------
# 2. Функция "Ворот с Задержкой" (Концептуально)
# -------------------------------------------
# Вместо прямого использования задержки в карте (что изменит ее свойства),
# мы можем думать о K*sin(...) как о нелинейном "вороте",
# а Omega как о "входном сигнале со сдвигом/задержкой" относительно
# основного такта итераций.

# -------------------------------------------
# 3. Вычисление Числа Вращения ("Считывание")
# -------------------------------------------
# Число вращения - среднее изменение фазы за одну итерацию.
# Аналог отношения частоты отклика системы к частоте драйвера (f_exp/fm в статье)

def calculate_winding_number(Omega, K, num_iterations=2000, transient=500):
    """Вычисляет число вращения для заданных Omega и K."""
    theta = 0.5 # Начальная фаза
    total_phase_change = 0.0
    theta_unwrapped = theta # Фаза без взятия по модулю 1, для отслеживания полных оборотов

    # Пропускаем переходный процесс
    for _ in range(transient):
        theta_next_raw = theta + Omega + (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)
        theta = theta_next_raw % 1

    # Вычисляем число вращения на установившемся режиме
    theta_unwrapped = theta # Сбрасываем после транзиента
    for _ in range(num_iterations - transient):
        theta_next_raw = theta + Omega + (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)
        total_phase_change += theta_next_raw - theta # Насколько фаза изменилась *до* взятия по модулю
        theta = theta_next_raw % 1

    winding_number = total_phase_change / (num_iterations - transient)
    # Или можно посчитать как (theta_final_unwrapped - theta_initial_unwrapped) / num_iterations
    # Альтернативный способ (менее подвержен ошибкам округления для долгих симуляций):
    # theta = 0.5
    # thetas_unwrapped = np.zeros(num_iterations)
    # current_unwrapped_theta = theta
    # for i in range(num_iterations):
    #     if i < transient:
    #          theta_next_raw = theta + Omega + (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)
    #          theta = theta_next_raw % 1
    #     else:
    #          theta_next_raw = theta + Omega + (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)
    #          current_unwrapped_theta += theta_next_raw - theta # Add the change before modulo
    #          theta = theta_next_raw % 1
    #          thetas_unwrapped[i] = current_unwrapped_theta

    # if num_iterations > transient:
    #      winding_number = (thetas_unwrapped[-1] - thetas_unwrapped[transient]) / (num_iterations - transient)
    # else:
    #      winding_number = 0


    return winding_number

# -------------------------------------------
# 4. Визуализация Лестницы Дьявола
# -------------------------------------------
K_devils_staircase = 1.0 # Сила связи (попробуйте > 1 для хаоса и перекрытия языков)
omega_values = np.linspace(0, 1, 400) # Диапазон параметра Omega
winding_numbers = []

print("Вычисление чисел вращения (может занять время)...")
for omega in omega_values:
    wn = calculate_winding_number(omega, K_devils_staircase, num_iterations=1500, transient=500)
    winding_numbers.append(wn)
print("Вычисление завершено.")

plt.figure(figsize=(10, 7))
plt.plot(omega_values, winding_numbers, '.', markersize=2, color='blue')
plt.xlabel("Параметр $\Omega$ (Аналог $f_0 / f_m$)")
plt.ylabel("Число вращения $W$ (Аналог $f_{exp} / f_m$)")
plt.title(f"Лестница Дьявола для Круговой Карты (K = {K_devils_staircase})\n(Аналогия с захватом частоты во временном кристалле)")
plt.grid(True, linestyle=':', alpha=0.7)

# Добавим линии для рациональных чисел вращения (плато)
rational_plateaus = {1/2: '1/2', 1/3: '1/3', 2/3: '2/3', 1/4: '1/4', 3/4: '3/4',
                     1/5: '1/5', 2/5: '2/5', 3/5: '3/5', 4/5: '4/5'}
for plateau_val, label in rational_plateaus.items():
    # Найдем Омега, где число вращения близко к плато
    indices = np.where(np.abs(np.array(winding_numbers) - plateau_val) < 1e-3)[0]
    if len(indices) > 1 : # Если есть плато
         plt.hlines(plateau_val, omega_values[indices[0]], omega_values[indices[-1]], color='red', lw=1.5)
         # Попытка разместить текст над плато
         text_x = omega_values[indices[0]] + (omega_values[indices[-1]] - omega_values[indices[0]]) / 2
         plt.text(text_x, plateau_val + 0.01, label, horizontalalignment='center', color='red')


plt.show()


# -------------------------------------------
# 5. Визуализация Динамики ("Разворот вычислений" во времени)
# -------------------------------------------
def plot_dynamics(Omega, K, num_iterations=100, title_suffix=""):
    """Визуализирует фазу системы со временем."""
    theta = 0.5
    thetas = np.zeros(num_iterations)
    for i in range(num_iterations):
        thetas[i] = theta
        theta = circle_map_step(theta, Omega, K)

    plt.figure(figsize=(10, 4))
    plt.plot(range(num_iterations), thetas, '.-', markersize=4)
    plt.xlabel("Итерация (Время t)")
    plt.ylabel("Фаза $\\theta$")
    wn = calculate_winding_number(Omega, K, 500, 100) # Приблизительно для заголовка
    plt.title(f"Динамика Фазы: $\Omega={Omega:.3f}, K={K}$ (W ≈ {wn:.3f}) {title_suffix}")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

# Примеры для разных режимов
print("\nВизуализация динамики для разных Omega:")
# Пример 1: Захват частоты 1/2 (на плато)
omega_locked_1_2 = 0.5 # Примерное значение из графика, найдите точнее, если нужно
plot_dynamics(omega_locked_1_2, K_devils_staircase, num_iterations=50, title_suffix="- Режим захвата 1/2 (Период 2)")

# Пример 2: Квазипериодический режим (между плато)
omega_quasi = 0.618 # Золотое сечение, часто приводит к квазипериодичности
plot_dynamics(omega_quasi, K_devils_staircase, num_iterations=100, title_suffix="- Квазипериодический режим")

# Пример 3: Если K > 1, можно показать хаос
K_chaos = 1.5
omega_chaos_example = 0.5
# plot_dynamics(omega_chaos_example, K_chaos, num_iterations=200, title_suffix="- Хаотический режим (K>1)") # Закомментировано, K=1 в основном примере

