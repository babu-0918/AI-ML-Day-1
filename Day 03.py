import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load and preprocess dataset
# Assuming dataset is in a CSV file
df = pd.read_csv("Housing.csv")  # Modify the filename if needed
df.dropna(inplace=True)  # Handling missing values

X = df[['Feature1', 'Feature2']]  # Replace with actual feature names
y = df['Price']  # Target variable

# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Fit Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# Step 5: Plot regression line (for simple linear regression)
plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
plt.scatter(X_test.iloc[:, 0], y_pred, color='red', label='Predicted')
plt.plot(X_test.iloc[:, 0], model.predict(X_test), color='green', linewidth=2)
plt.xlabel("Feature")
plt.ylabel("Price")
plt.legend()
plt.show()

# Interpretation of coefficients
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
