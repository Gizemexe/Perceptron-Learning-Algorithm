import pandas as pd
import numpy as np

# Load the data
train_data = pd.read_excel("DataForPerceptron.xlsx", sheet_name="TRAINData")
test_data = pd.read_excel("DataForPerceptron.xlsx", sheet_name="TESTData")

# Define input and output features
X_train = train_data.iloc[:, 1:-1].values  # Özellikler
y_train = np.where(train_data.iloc[:, -1].values == 2, -1, 1)  # Sınıfları 1 ve -1 olarak dönüştür.
X_test = test_data.iloc[:, 1:-1].values  # Test özellikleri


y_test_original = test_data.iloc[:, -1].values
y_test = np.where(y_test_original == 2, -1, 1)  # 2 için -1, 4 için 1 olarak dönüştür.

# Perceptron algorithm
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Ağırlık ve bias
        self.weights = np.zeros(n_features)
        self.bias = 0


        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Update weights and bias based on error
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, -1)

# Train the model and make predictions
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X_train, y_train)

# Predict on test data
test_predictions = perceptron.predict(X_test)

# Predict on train data
train_predictions = perceptron.predict(X_train)

# (1 -> class 4, -1 -> class 2)
predicted_classes = np.where(test_predictions == 1, 4, 2)

# Calculate accuracy
accuracy = np.mean(test_predictions == y_test) * 100
train_accuracy = np.mean(train_predictions == y_train) * 100

# Results
results_df = pd.DataFrame({
    "Test Sample": np.arange(1, len(predicted_classes) + 1),
    "Predicted Class": predicted_classes,
    "Actual Class": y_test_original  # Actual class from the test data
})

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Training Accuracy: {train_accuracy:.2f}%")
print("\nClassification Results:")
print(results_df)
