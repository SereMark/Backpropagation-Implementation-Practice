"""
Neural Network Backpropagation Implementation

This script is part of an educational project designed to help learners understand and implement backpropagation in a simple neural network. 

💡 For detailed instructions, explanations, and examples, please refer to the README.md file in this repository.

Project Summary:
- Train a neural network to approximate the XOR function.
- Complete the `backward` method to implement backpropagation.
- Test the network's performance after training.

"""

import math
import random

# Sigmoid activation function: helps the network make predictions between 0 and 1
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid: used to compute how much each neuron contributes to the error
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Mean squared error: measures how far predictions are from actual values
def mean_squared_error(y_true, y_pred):
    return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)

# Dot product: multiplies corresponding elements of two lists and sums them up
def dot_product(vector1, vector2):
    return sum(a * b for a, b in zip(vector1, vector2))

# A simple Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the neural network with random weights and zero biases.
        - input_size: Number of input values (e.g., 2 for XOR inputs like [0, 1])
        - hidden_size: Number of neurons in the hidden layer
        - output_size: Number of output values (e.g., 1 for XOR outputs like [1])
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases from input to hidden layer
        self.weights_input_to_hidden = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.bias_hidden = [0 for _ in range(hidden_size)]

        # Weights and biases from hidden to output layer
        self.weights_hidden_to_output = [[random.uniform(-0.1, 0.1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.bias_output = [0 for _ in range(output_size)]

    def forward(self, inputs):
        """
        Performs a forward pass through the network.
        - inputs: A list of input values (e.g., [0, 1])
        Returns the output of the network.
        """
        # Step 1: Compute hidden layer input and output
        hidden_layer_input = [dot_product(inputs, weights) + bias for weights, bias in zip(zip(*self.weights_input_to_hidden), self.bias_hidden)]
        hidden_layer_output = [sigmoid(x) for x in hidden_layer_input]

        # Step 2: Compute output layer input and output
        output_layer_input = [dot_product(hidden_layer_output, weights) + bias for weights, bias in zip(zip(*self.weights_hidden_to_output), self.bias_output)]
        output_layer_output = [sigmoid(x) for x in output_layer_input]

        # Save values for backward pass
        self.inputs = inputs
        self.hidden_layer_output = hidden_layer_output
        self.output_layer_output = output_layer_output

        return output_layer_output

    def backward(self, targets, learning_rate):
        """
        Performs a backward pass through the network.
        - targets: Actual output values (e.g., [1] for XOR)
        - learning_rate: Step size for weight updates
        """
        # Step 1: Calculate the error at the output layer
        # Step 2: Compute gradients for weights and biases at the output layer
        # Step 3: Propagate the error back to the hidden layer
        # Step 4: Compute gradients for weights and biases at the hidden layer
        # Step 5: Update weights and biases for both layers
        pass

    def train(self, inputs_list, targets_list, epochs, learning_rate):
        """
        Trains the network for a specified number of epochs.
        - inputs_list: List of input data
        - targets_list: Corresponding list of target outputs
        - epochs: Number of training iterations
        - learning_rate: Step size for weight updates
        """
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in zip(inputs_list, targets_list):
                # Forward pass
                self.forward(inputs)

                # Backward pass
                self.backward(targets, learning_rate)

                # Calculate loss
                total_loss += mean_squared_error(targets, self.output_layer_output)

            # Print average loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Average Loss = {total_loss / len(inputs_list):.4f}")

if __name__ == "__main__":
    print("Welcome to the Neural Network Backpropagation Implementation!")
    print("👉 This script will train a simple neural network on the XOR function.")
    print("💡 To learn more about the project or complete your tasks, please refer to the README.md.")
    print("\nTraining begins...\n")
    
    # XOR dataset
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]

    # Define network parameters
    input_size = 2
    hidden_size = 2
    output_size = 1
    learning_rate = 0.1
    epochs = 1000

    # Create and train the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(inputs, targets, epochs, learning_rate)

    # Test the network
    print("\nTesting...")
    for input_vector, target in zip(inputs, targets):
        prediction = nn.forward(input_vector)
        print(f"Input: {input_vector} - Predicted: {[round(p, 2) for p in prediction]} - Target: {target}")
