# 4 (a) 1.

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['target'] = iris.target

print(df.head())

# 4 (a) 2.
X = iris.data  # Features (Sepal Length, Sepal Width, Petal Length, Petal Width)
y = iris.target  # Class labels (0, 1, 2)

# Print the shape of X and y
print("Feature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)


from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


"""I've first defined all functions, then executed (called them) all together at last"
# 4 (b) 1.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

print("Training Set Shape:", X_train.shape)
print("Test Set Shape:", X_test.shape)


# 4 (b) 2. 

# Normalizing
import numpy as np

def min_max_scaling(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

X_train_norm = min_max_scaling(X_train)
X_test_norm = min_max_scaling(X_test)

# Standardizing

def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_train_std = standardize(X_train)
X_test_std = standardize(X_test)

# No processing

X_train_raw = X_train.copy()
X_test_raw = X_test.copy()

# 4. (c)

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


# 4. (d) 1.

def predict_proba(X, W, b):
    logits = np.dot(X, W) + b
    return softmax(logits)

# 4. (d) 2.

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]  
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m 

encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))
y_train = encoder.fit_transform(y_train.reshape(-1, 1))  
y_test = encoder.transform(y_test.reshape(-1, 1))

# 4. (d) 3. 1.

def train_gradient_descent(X_train, y_train, lr=0.1, epochs=1000):
   
    m, n = X_train.shape  # m = samples, n = features
    k = y_train.shape[1]  # Number of classes
    W = np.zeros((n, k))  # Initialize weights
    b = np.zeros((1, k))  # Initialize biases

    loss_history = []

    for epoch in range(epochs):
        logits = np.dot(X_train, W) + b
        y_pred = softmax(logits)
        loss = cross_entropy_loss(y_train, y_pred)

        grad_W = np.dot(X_train.T, (y_pred - y_train)) / m
        grad_b = np.sum(y_pred - y_train, axis=0, keepdims=True) / m

        W -= lr * grad_W
        b -= lr * grad_b

        loss_history.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return W, b, loss_history

def train_stochastic_gradient_descent(X_train, y_train, lr=0.1, epochs=1000):
  
    m, n = X_train.shape  # m = samples, n = features
    k = y_train.shape[1]  # Number of classes
    W = np.zeros((n, k))  # Initialize weights
    b = np.zeros((1, k))  # Initialize biases

    loss_history = []

    for epoch in range(epochs):
        idx = np.random.randint(m)  # Pick one sample randomly
        X_sample, y_sample = X_train[idx:idx+1], y_train[idx:idx+1]

        logits = np.dot(X_sample, W) + b
        y_pred = softmax(logits)
        loss = cross_entropy_loss(y_sample, y_pred)

        grad_W = np.dot(X_sample.T, (y_pred - y_sample))
        grad_b = np.sum(y_pred - y_sample, axis=0, keepdims=True)

        W -= lr * grad_W
        b -= lr * grad_b

        loss_history.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return W, b, loss_history

# 4. (d) 4.

def evaluate_model(X_test, y_test, W, b):
    \
    y_pred_probs = softmax(np.dot(X_test, W) + b)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=iris.target_names)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)

    return accuracy, conf_matrix

# 4. (d) 5.

print("\nTraining using Full-Batch Gradient Descent")
W_gd, b_gd, loss_gd = train_gradient_descent(X_train, y_train, lr=0.1, epochs=1000)


print("\nTraining using Stochastic Gradient Descent")
W_sgd, b_sgd, loss_sgd = train_stochastic_gradient_descent(X_train, y_train, lr=0.1, epochs=1000)

print("\nFull-Batch Gradient Descent Results:")
evaluate_model(X_test, y_test, W_gd, b_gd)

print("\nStochastic Gradient Descent (SGD) Results:")
evaluate_model(X_test, y_test, W_sgd, b_sgd)


plt.plot(loss_gd, label="Full-Batch GD")
plt.plot(loss_sgd, label="SGD", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()


print("\nResults with Normalization:")
W_norm, b_norm, _ = train_gradient_descent(X_train_norm, y_train)
evaluate_model(X_test_norm, y_test, W_norm, b_norm)

print("\nResults with Standardization:")
W_std, b_std, _ = train_gradient_descent(X_train_std, y_train)
evaluate_model(X_test_std, y_test, W_std, b_std)

print("\nResults without Processing:")
W_raw, b_raw, _ = train_gradient_descent(X_train, y_train)
evaluate_model(X_test, y_test, W_raw, b_raw)



plt.figure(figsize=(10, 5))
plt.plot(loss_gd, label="Full-Batch GD", color="blue")
plt.plot(loss_sgd, label="SGD", linestyle="dashed", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs: Full-Batch GD vs SGD")
plt.legend()
plt.grid(True)
plt.show()

import seaborn as sns
def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    """Plots a heatmap for the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

_, conf_matrix_gd = evaluate_model(X_test, y_test, W_gd, b_gd)
_, conf_matrix_sgd = evaluate_model(X_test, y_test, W_sgd, b_sgd)


plot_confusion_matrix(conf_matrix_gd, title="Confusion Matrix: Full-Batch Gradient Descent")
plot_confusion_matrix(conf_matrix_sgd, title="Confusion Matrix: Stochastic Gradient Descent")


