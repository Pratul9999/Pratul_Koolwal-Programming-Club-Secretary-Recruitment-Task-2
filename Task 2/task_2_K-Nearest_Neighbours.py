# This model first creates a dataset using Mean, standard deviation, Upper and lower bounds, and a correlation matrix.
# Then it uses K-Nearest Neighbours (from library sci-kit learn) to predict lung tidal volume using this dataset.

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split

params = {
    'Weight': {'mean': 65.54, 'std': 12.80, 'lower': 38, 'upper': 107},
    'Height': {'mean': 161.50, 'std': 9.00, 'lower': 138, 'upper': 189},
    'Chest_Circumference': {'mean': 94.19, 'std': 10.55, 'lower': 61, 'upper': 130},
    'BMI': {'mean': 25.04, 'std': 3.93, 'lower': 15.99, 'upper': 37.33},
    'Age': {'mean': 53.86, 'std': 16.62, 'lower': 12, 'upper': 98},
    'Tidal_Volume': {'mean': 475.87, 'std': 66.61, 'lower': 325, 'upper': 675}
}

correlation_matrix = np.array([
    [1.0, 0.5, 0.3, 0.7, 0.6, 0.825],
    [0.5, 1.0, 0.4, 0.8, 0.2, 0.880],
    [0.3, 0.4, 1.0, 0.5, 0.9, 0.889],
    [0.7, 0.8, 0.5, 1.0, 0.9, 0.905],
    [0.6, 0.2, 0.9, 0.9, 1.0, 0.910],
    [0.825, 0.880, 0.889, 0.905, 0.910, 1.0],
])

n = 1000

means = np.array([params[p]['mean'] for p in params])
stds = np.array([params[p]['std'] for p in params])

covariance_matrix = np.outer(stds, stds) * correlation_matrix

data = np.random.multivariate_normal(means, covariance_matrix, n)

def truncate_data(data, param_info):
    truncated_data = np.copy(data)
    for i, param in enumerate(param_info):
        lower, upper = param_info[param]['lower'], param_info[param]['upper']
        truncated_data[:, i] = np.clip(data[:, i], lower, upper)
    return truncated_data

truncated_data = truncate_data(data, params)

dataset = pd.DataFrame(truncated_data, columns=params.keys())


X = dataset[['Weight', 'Height', 'Chest_Circumference', 'BMI', 'Age']]  # Example features
y = dataset['Tidal_Volume']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsRegressor

k = 5 
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

y_pred = knn_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Tidal Volume')
plt.ylabel('Predicted Tidal Volume')
plt.title('Actual vs Predicted Tidal Volume')
plt.show()
