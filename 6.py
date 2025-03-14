# 6 (a) 1
import numpy as np

np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)


# 6 (a) 2
import matplotlib.pyplot as plt

plt.hist(data, bins=30, density=True, alpha=0.6, color='b')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram of Simulated Normal Data')
plt.show()


# 6 (b) 1
mu_hat = np.mean(data)
sigma_hat = np.std(data, ddof=0)
(mu_hat, sigma_hat)


# 6 (b) 2
from scipy.stats import norm

x = np.linspace(min(data), max(data), 100)
pdf = norm.pdf(x, mu_hat, sigma_hat)

plt.hist(data, bins=30, density=True, alpha=0.6, color='b', label="Histogram")
plt.plot(x, pdf, 'r', label="Fitted Normal PDF")
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram with Fitted Normal Distribution')
plt.legend()
plt.show()


# 6 (c) 1
outliers = np.random.uniform(100, 150, 50)
data_with_outliers = np.concatenate([data, outliers])


# 6 (c) 2
mu_hat_outliers = np.mean(data_with_outliers)
sigma_hat_outliers = np.std(data_with_outliers, ddof=0)

x = np.linspace(min(data_with_outliers), max(data_with_outliers), 100)
pdf_outliers = norm.pdf(x, mu_hat_outliers, sigma_hat_outliers)

plt.hist(data_with_outliers, bins=30, density=True, alpha=0.6, color='b', label="Histogram with Outliers")
plt.plot(x, pdf_outliers, 'r', label="Fitted Normal PDF with Outliers")
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Effect of Outliers on Normal Distribution Fitting')
plt.legend()
plt.show()


# 6 (c) 3
print("Original Mean and Std Dev:", mu_hat, sigma_hat)
print("With Outliers Mean and Std Dev:", mu_hat_outliers, sigma_hat_outliers)

z_scores = (data_with_outliers - mu_hat_outliers) / sigma_hat_outliers
outlier_indices = np.where(np.abs(z_scores) > 3)
outlier_values = data_with_outliers[outlier_indices]

print("Detected Outliers:", outlier_values)

"""Mean shifts upwards, standard deviation increases
Approach to handle outliers: Z score formula and Inter Quartile range method"""
