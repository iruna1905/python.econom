# Множинна регресія
from colorama import Fore
import numpy as np
from sklearn.linear_model import LinearRegression

print(f"9-й варіант")

# print("Вводимо дані")
y = [70.1, 71.3, 75, 80.1, 81.2, 85.6, 90, 90.5, 94.6, 95, 100.2, 106.8,
     109.3, 112.3, 120.8, 125.7, 124.2]

x = [
    [56.2, 1.85], [58.6, 2.35], [60, 2.5], [64.2, 2.6], [66.2, 2.6],
    [68.3, 2.9], [69, 2.85], [72.1, 3.1], [74.3, 3.3], [75.5, 2.95],
    [74.5, 3.25], [76.6, 3.4], [79.8, 3.15], [77.7, 3.55], [5.00, 47.84],
    [80.2, 3.56], [79.2, 3.65], [81.2, 4]
]
x, y = np.array(x), np.array(y)

# print("Створення самої моделі:")
model = LinearRegression().fit(x, y)
print(model)

# print("Отримаємо результати:")
# R² = (score)
# round(), 4 округлення
r_sq = round(model.score(x, y), 4)
print(f"Коефіцієнт детермінації (𝑅²): {r_sq}")

print(f"intercept (a, 𝑏₀): {round(model.intercept_, 4)}")
print(f"slope (b, 𝑏₁): {model.coef_}")

# print("Прогнозуємо відповідь")
# g(xi) задяки функції Python
y_pred = model.predict(x)
print(f"predicted response (g(xi)): \n{y_pred}")

# g(xi) по формулі
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response (g(xi)):\n{y_pred}")

# модель для нових даних
x_new = np.arange(10).reshape((-1, 2))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

print(Fore.MAGENTA + f"Висновки:"
                     f"\n (𝑅²): {r_sq},"
                     f"\n (a, 𝑏₀): {round(model.intercept_, 4)},"
                     f"\n (b, 𝑏₁): {model.coef_}")
