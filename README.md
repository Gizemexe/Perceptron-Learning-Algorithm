## Perceptron Algorithm 
In this report, I present the implementation and evaluation of a Perceptron algorithm for the binary classification task.The goal is train a Perceptron model using the provided training data, test it on a separate test dataset and report on the model's performance.

## Dataset
The data set has two pages:
- TRAINData: Contains the training samples with their respective features and class labels.
- TESTData: Contains the test samples with features and actual class labels for evaluation.
The dataset contains input features and their corresponding class labels (2 and 4), which I mapped to values of -1 and 1 respectively for classification purposes.

## Model Implementation
The Perceptron algorithm is a linear classifier that iteratively adjusts its weights based on the misclassified examples [1]. The model uses the following approach:
- Initialization: The model starts with zero weights and bias.
- Learning Rate: 0.01.
- Iterations: Trained for 1000 iterations or epochs.
- Activation Function: A step function that returns 1 if the input is greater than or equal to zero, and -1 otherwise.

## Test Results
<p>After training, I tested the model on the test dataset (TESTData) and compared the predicted class labels with the actual class labels to evaluate the performance of the model.</p>

**The Output:**

![image](https://github.com/user-attachments/assets/be0f66ad-605f-4b8d-b74b-432da9c2ce84)

![image](https://github.com/user-attachments/assets/da45ebd6-5d7d-4d16-a5ef-005d2e87463a)

<p>Note: The Actual Class column represents the class labels from the test data. It shows the initial test data content.</p>

## Training Accuracy
<p>To further evaluate the model, the training accuracy was also calculated. This provides an indication of how well the model performed on the data it was trained on.</p>

**Calculate training accuracy**

train_predictions = perceptron.predict(X_train) …

train_accuracy = np.mean(train_predictions == y_train) * 100

…

print(f"Training Accuracy: {train_accuracy:.2f}%")


<p>The training accuracy was found to be 94.18%. You can see the screenshot in “Test Results” section.</p>

## Conclusion
<p>The Perceptron algorithm was successfully implemented and tested on the given dataset. The test accuracy of the model is quite low at 25.56%, suggesting that a more complex model or tuning of the Perceptron parameters might be necessary to improve performance.</p>

## References
[1] Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
