# Розширина регресія
from colorama import Fore
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

print(f"9-й варіант")

# print(f"даємо та трансформуємо дані")
y = [70.1, 71.3, 75, 80.1, 81.2, 85.6, 90, 90.5, 94.6, 95, 100.2, 106.8,
     109.3, 112.3, 120.8, 125.7, 124.2]

x = [
    [56.2, 1.85],  [58.6, 2.35],  [60, 2.5],     [64.2, 2.6], [66.2, 2.6],
    [68.3, 2.9],   [69, 2.85],    [72.1, 3.1],   [74.3, 3.3], [75.5, 2.95],
    [74.5, 3.25],  [76.6, 3.4],   [79.8, 3.15],  [77.7, 3.55], [5.00, 47.84],
    [80.2, 3.56],  [79.2, 3.65],  [81.2, 4]
]
x, y = np.array(x), np.array(y)

x = sm.add_constant(x)
print(f"X = \n{x}")
print(f"Y = \n{y}")

# print("Створення самої моделі:")
model: OLS = sm.OLS(y, x)
results = model.fit()

# Regression
print(Fore.MAGENTA, f"Regression (Регресія) як в Exel:"
                    f"\n{results.summary()}")
