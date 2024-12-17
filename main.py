import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load datasets
exercise_data = pd.read_csv("exercise.csv")
calories_data = pd.read_csv("calories.csv")

# Merge datasets
data = pd.concat([exercise_data, calories_data['Calories']], axis=1)

# Explore data
print(data.info())
print(data.describe())
print(data.isnull().sum())
data.replace({'Gender': {'male': 0, 'female': 1}}, inplace=True)


# Visualize data
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()

# Preprocess data
data.replace({'Gender': {'male': 0, 'female': 1}}, inplace=True)
X = data.drop(columns=['Calories'])
Y = data['Calories']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, Y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(Y_test, predictions)
print("Mean Absolute Error:", mae)

# Visualize results
plt.figure(figsize=(10, 5))
plt.plot(Y_test.values[:50], label="Actual")
plt.plot(predictions[:50], label="Predicted")
plt.legend()
plt.show()
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load datasets
exercise_data = pd.read_csv("exercise.csv")
calories_data = pd.read_csv("calories.csv")

# Merge datasets
data = pd.concat([exercise_data, calories_data['Calories']], axis=1)

# Explore data
print(data.info())
print(data.describe())
print(data.isnull().sum())


# Visualize data
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()

# Preprocess data
data.replace({'Gender': {'male': 0, 'female': 1}}, inplace=True)
X = data.drop(columns=['Calories'])
Y = data['Calories']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, Y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(Y_test, predictions)
print("Mean Absolute Error:", mae)

# Visualize results
plt.figure(figsize=(10, 5))
plt.plot(Y_test.values[:50], label="Actual")
plt.plot(predictions[:50], label="Predicted")
plt.legend()
plt.show()
