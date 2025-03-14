# 5 (a) 1
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
df = pd.read_excel(url)

# 5 (a) 2
df.head()

# 5 (a) 3
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 5 (a) 4
df.isnull().sum()

# 5 (a) 5
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5 (b) 1
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

# 5 (b) 2
from sklearn.metrics import mean_squared_error, r2_score

y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2

# 5 (b) 3
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs. Actual Values")
plt.show()


# 5 (c) 1
from sklearn.preprocessing import PolynomialFeatures

poly2 = PolynomialFeatures(degree=2)
poly3 = PolynomialFeatures(degree=3)
poly4 = PolynomialFeatures(degree=4)

X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)

X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)

X_train_poly4 = poly4.fit_transform(X_train)
X_test_poly4 = poly4.transform(X_test)


# 5 (c) 2
lr2 = LinearRegression()
lr3 = LinearRegression()
lr4 = LinearRegression()

lr2.fit(X_train_poly2, y_train)
lr3.fit(X_train_poly3, y_train)
lr4.fit(X_train_poly4, y_train)

# 5 (c) 3
y_pred2 = lr2.predict(X_test_poly2)
y_pred3 = lr3.predict(X_test_poly3)
y_pred4 = lr4.predict(X_test_poly4)

mse2 = mean_squared_error(y_test, y_pred2)
mse3 = mean_squared_error(y_test, y_pred3)
mse4 = mean_squared_error(y_test, y_pred4)

r2_2 = r2_score(y_test, y_pred2)
r2_3 = r2_score(y_test, y_pred3)
r2_4 = r2_score(y_test, y_pred4)

mse2, mse3, mse4, r2_2, r2_3, r2_4

# 5 (c) 4
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Polynomial Degree 2")

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred3)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Polynomial Degree 3")

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred4)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Polynomial Degree 4")

plt.tight_layout()
plt.show()


# 5 (d) 1
import numpy as np

plt.scatter(y_test, y_pred, label="Linear Regression", alpha=0.6)
plt.scatter(y_test, y_pred2, label="Polynomial Degree 2", alpha=0.6)
plt.scatter(y_test, y_pred3, label="Polynomial Degree 3", alpha=0.6)
plt.scatter(y_test, y_pred4, label="Polynomial Degree 4", alpha=0.6)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Comparison of Regression Models")
plt.legend()
plt.show()



# 5 (d) 2
models = ["Linear", "Poly 2", "Poly 3", "Poly 4"]
mse_values = [mse, mse2, mse3, mse4]
r2_values = [r2, r2_2, r2_3, r2_4]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(models, mse_values, color=["blue", "green", "orange", "red"])
plt.xlabel("Model")
plt.ylabel("Mean Squared Error")
plt.title("MSE Comparison")

plt.subplot(1, 2, 2)
plt.bar(models, r2_values, color=["blue", "green", "orange", "red"])
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("R² Score Comparison")

plt.tight_layout()
plt.show()


# 5 (e) 1
bias_variance_analysis = {
    "Linear Regression": "High bias, low variance",
    "Polynomial Degree 2": "Moderate bias, moderate variance",
    "Polynomial Degree 3": "Lower bias, higher variance",
    "Polynomial Degree 4": "Very low bias, very high variance (overfitting)"
}

bias_variance_analysis

"""Linear regression suffers from underfitting due to high bias.
Higher-degree polynomial models capture more complexity but risk overfitting due to high variance.
Polynomial regression with degree 2 or 3 often provides the best balance between bias and variance, giving good predictive performance without excessive complexity."""

# 5 (e) 2
overfitting_explanation = """
Higher-degree polynomial models tend to overfit the data because they introduce more parameters, allowing the model to fit noise in the training set
instead of capturing general patterns. This leads to high variance, meaning the model performs well on training data but poorly on unseen test data. Overfitting
occurs when the model is too complex relative to the amount of data available, making it highly sensitive to small fluctuations in training data.
As a result, its predictions on new data are unreliable. Regularization techniques (like Lasso or Ridge regression) and cross-validation can help mitigate 
overfitting by penalizing excessive complexity.
"""
overfitting_explanation

