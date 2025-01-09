import numpy as np
import scipy as sc

def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Матрицы не согласованы для умножения.")

    rows, cols, common_dim = len(matrix_a), len(matrix_b[0]), len(matrix_b)
    result_matrix = [[0] * cols for _ in range(rows)]

    for row_idx in range(rows):
        for col_idx in range(cols):
            result_matrix[row_idx][col_idx] = sum(
                matrix_a[row_idx][k] * matrix_b[k][col_idx] for k in range(common_dim)
            )

    return result_matrix

def functions(coeffs_1, coeffs_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coefficients_1 = list(map(int, coeffs_1.split()))
    coefficients_2 = list(map(int, coeffs_2.split()))

    if coefficients_1 == coefficients_2:
        return None

    delta = (coefficients_1[1] - coefficients_2[1])**2 - 4 * (coefficients_1[0] - coefficients_2[0]) * (coefficients_1[2] - coefficients_2[2])

    if coefficients_1[0] == coefficients_2[0]:
        if coefficients_1[1] == coefficients_2[1]:
            return []
        x = -(coefficients_1[2] - coefficients_2[2]) / (coefficients_1[1] - coefficients_2[1])
        y = coefficients_1[0] * x**2 + coefficients_1[1] * x + coefficients_1[2]
        return [(round(x, 3), round(y, 3))]

    if delta < 0:
        return []

    sqrt_delta = delta**0.5
    factor = 2 * (coefficients_1[0] - coefficients_2[0])

    x1 = (-(coefficients_1[1] - coefficients_2[1]) + sqrt_delta) / factor
    x2 = (-(coefficients_1[1] - coefficients_2[1]) - sqrt_delta) / factor

    roots = [
        (round(x1, 3), round(coefficients_1[0] * x1**2 + coefficients_1[1] * x1 + coefficients_1[2], 3)),
        (round(x2, 3), round(coefficients_1[0] * x2**2 + coefficients_1[1] * x2 + coefficients_1[2], 3))
    ]

    return roots if delta > 0 else roots[:1]

def compute_moments(data):
    n = len(data)
    mean_value = sum(data) / n

    central_moments = {
        2: sum((x - mean_value)**2 for x in data) / n,
        3: sum((x - mean_value)**3 for x in data) / n,
        4: sum((x - mean_value)**4 for x in data) / n,
    }

    skewness = central_moments[3] / central_moments[2]**1.5
    kurtosis = central_moments[4] / central_moments[2]**2 - 3

    return skewness, kurtosis

def skew(data):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    return round(compute_moments(data)[0], 2)

def kurtosis(data):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    return round(compute_moments(data)[1], 2)
