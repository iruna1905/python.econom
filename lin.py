# Лінійна регресія
from colorama import Fore
import numpy as np
from sklearn.linear_model import LinearRegression

print(f"9-й варіант")

# print("Вводимо дані")
print(f"Подамо x як двовимірний масив:")
x = np.array([10.2, 11.3, 14.6, 15.2, 17.2, 19.6, 20.3, 23.5, 25.6, 28.9]).reshape((-1, 1))
y = np.array([7.8, 11.2, 14.6, 14.9, 16.9, 18.1, 17.5, 22.5, 25.8, 26.2])

# print("Створення самої моделі:")
model = LinearRegression()
print(model.fit(x, y))

# print("Отримаємо результати:")
# R² = (score)
# round(), 4 округлення
r_sq = round(model.score(x, y), 4)
print(f"Коефіцієнт детермінації (𝑅²): {r_sq}")

print(f"intercept (a, 𝑏₀): {round(model.intercept_, 4)}")
print(f"slope (b, 𝑏₁): {model.coef_}")

print(f"Подамо і y як двовимірний масив:")

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept (a, 𝑏₀): {new_model.intercept_}")
print(f"slope (b, 𝑏₁): {new_model.coef_}")

# print("Прогнозуємо відповідь")
# g(xi) задяки функції Python
y_pred = model.predict(x)
print(f"predicted response (g(xi)): \n{y_pred}")

# g(xi) по формулі
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response (g(xi)):\n{y_pred}")

# модель для нових даних
x_new = np.arange(5).reshape((-1, 1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

print(Fore.MAGENTA + f"Висновки:"
                     f"\n (𝑅²): {r_sq},"
                     f"\n (a, 𝑏₀): {round(model.intercept_, 4)},"
                     f"\n (b, 𝑏₁): {model.coef_}")
