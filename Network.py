import numpy as np
from sklearn.preprocessing import StandardScaler

class Layer:
    def __init__(self, output_size, activation=None, dropout_rate=0.0):
        self.output_size = output_size  # Neurons in layer
        self.W = None
        self.b = None
        self.inputs = None
        self.activation = activation  # Activation functions: None for linear, sigmoid or relu
        self.dropout_rate = dropout_rate
        self.dropout_mask = None  # Required for saving the dropped neurons

    def initialize_weights(self, input_size):
        xavier_stddev = np.sqrt(2 / (input_size + self.output_size))
        self.W = np.random.randn(input_size, self.output_size) * xavier_stddev
        self.b = np.random.randn(self.output_size) * xavier_stddev

    def forward(self, inputs, train=True):
        self.inputs = inputs
        input_size = inputs.shape[-1]
        # Weights and bias initialization
        if self.W is None:
            self.initialize_weights(input_size)

        linear_output = np.dot(inputs, self.W) + self.b

        # Dropout using binomial distribution
        if train and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=linear_output.shape)
            linear_output *= self.dropout_mask / (1 - self.dropout_rate)

        # Activation
        if self.activation is None:
            return linear_output
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-linear_output))
        elif self.activation == 'relu':
            return np.maximum(0, linear_output)
        else:
            raise ValueError("Supported activations: None for linear, 'sigmoid', 'relu'")

    def backward(self, grad_output, learning_rate):
        # Derivatives for backward
        if self.activation is None:
            activation_derivative = 1
        elif self.activation == 'sigmoid':
            sigmoid_output = 1 / (1 + np.exp(-self.inputs))
            activation_derivative = sigmoid_output * (1 - sigmoid_output)
        elif self.activation == 'relu':
            activation_derivative = np.where(self.inputs > 0, 1, 0)
        else:
            raise ValueError("Supported activations: None for linear, 'sigmoid', 'relu'")

        # Updating only the neurons left after the dropout
        if self.dropout_mask is not None:
            grad_output *= self.dropout_mask / (1 - self.dropout_rate)

        # Updating the parameters
        grad_inputs = np.dot(grad_output, self.W.T) * activation_derivative
        grad_W = np.dot(self.inputs.T, grad_output)
        grad_b = np.sum(grad_output, axis=0)

        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b

        return grad_inputs

class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs, train=True):
        # Forward propagation with recursive loop
        for layer in self.layers:
            inputs = layer.forward(inputs, train)
        return inputs

    def backward(self, grad_output, learning_rate):
        # Backward propagation with reversed recursive loop
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)
        return grad_output

    def call(self, inputs, targets=None, num_epochs=50000, learning_rate=1e-3, train=True, problem_type='regression'):

        if train:
            for epoch in range(num_epochs):
                outputs = self.forward(inputs, train)
                if problem_type == 'regression':  # MSE and gradient for the regression problem
                    loss = np.mean((outputs - targets) ** 2) / 2
                    grad_output = (outputs - targets) / len(targets)
                elif problem_type == 'classification':
                    loss = -np.mean(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))
                    grad_output = (-targets / outputs + (1 - targets) / (1 - outputs)) / len(targets)
                else:
                    raise ValueError("Invalid problem_type. Supported values: 'regression' or 'classification'")

                self.backward(grad_output, learning_rate)

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: Loss = {loss:.8f}")
        else:
            outputs = self.forward(inputs, train)
            return outputs

np.random.seed(42)
X = np.random.randn(300, 5)  # Generate 5-dimensional input data
y = np.random.randint(0, 2, size=(300,1))  # Generate binary labels (0 or 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets for classification
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Create and configure the network for classification
network_classification = Network()
network_classification.add_layer(Layer(output_size=5, activation='relu'))
network_classification.add_layer(Layer(output_size=10, activation='relu'))
network_classification.add_layer(Layer(output_size=1, activation='sigmoid'))

# Train the network for classification
network_classification.call(X_train, y_train, num_epochs=50000, learning_rate=0.001, train=True, problem_type='classification')

# Evaluate the network on the test set for classification
test_predictions = network_classification.call(X_test, train=False)

test_predictions = np.round(test_predictions >= 0.5).astype(int)
accuracy = np.mean(test_predictions == y_test)
print("Classification Accuracy:", accuracy)

'''

IMPORTANT: To check the code for the regression use the the snippet below:

np.random.seed(42)
X = np.random.randn(300, 5)  # Generate 5-dimensional input data for regression
y = np.tanh(np.sum(X, axis=1)) + np.random.normal(0, 0.5, size=(300,))
y = y.reshape(-1, 1)

scaler_regression = StandardScaler()
X = scaler_regression.fit_transform(X)

# Split data into train and test sets for regression
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Create and configure the network for regression
network_regression = Network()
network_regression.add_layer(Layer(output_size=5))
network_regression.add_layer(Layer(output_size=10))
network_regression.add_layer(Layer(output_size=1))

# Train the network for regression
network_regression.call(X_train, y_train, num_epochs=10000, learning_rate=1e-4, train=True)

# Evaluate the network on the test set for regression
test_predictions = network_regression.call(X_test, train=False)
test_loss = np.mean((test_predictions - y_test) ** 2) / 2
print("Regression Test Loss:", test_loss)

'''