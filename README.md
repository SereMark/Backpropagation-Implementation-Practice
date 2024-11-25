# Neural Network Backpropagation Implementation

This project implements a simple neural network for learning the XOR function using backpropagation. It is designed as an educational tool to help learners understand the backpropagation algorithm and its role in training neural networks.

---

## 📋 Project Structure

- **`neural_network.py`**: The main Python script that contains the implementation of the neural network, forward pass, and utility functions. The `backward` method is left incomplete for you to implement.

---

## 🚀 How to Use

### 1. Clone the Repository
```bash
git clone <repository-url>
cd neural-network-backprop
```

### 2. Run the Code
The provided code includes a partially implemented neural network. It uses the XOR dataset for training and testing. To run the code:
```bash
python neural_network.py
```

### 3. Implement Backpropagation
🚨 **Important:** To fully train the network, you need to complete the `backward` method within the `NeuralNetwork` class.

---

## 🛠️ Your Task: Implement Backpropagation

The `backward` method is currently a placeholder. Here's what you need to do:

1. **Calculate Output Error**:
   - Compute the difference between the network's predictions and the target values.

2. **Output Layer Gradients**:
   - Use the derivative of the sigmoid function to calculate the gradient for weights and biases in the output layer.

3. **Propagate Error to Hidden Layer**:
   - Use the output gradients to calculate how much each hidden neuron contributes to the overall error.

4. **Hidden Layer Gradients**:
   - Compute gradients for weights and biases in the hidden layer.

5. **Update Weights and Biases**:
   - Apply the gradients to update weights and biases using the learning rate.

---

## 📊 Dataset: XOR Function

The network is tested on the XOR problem:
- **Inputs**: 
  - `[0, 0]` -> Target: `0`
  - `[0, 1]` -> Target: `1`
  - `[1, 0]` -> Target: `1`
  - `[1, 1]` -> Target: `0`

- **Expected Output**: After training, the network should approximate the XOR function.

---

## 📚 Learning Objectives

1. Understand the structure of a simple neural network (input, hidden, and output layers).
2. Implement backpropagation by calculating and propagating errors through the network.
3. Learn how weights and biases are updated during training.
4. Explore how parameters like learning rate and hidden layer size affect training performance.

---

## ⚙️ Adjustable Parameters

The network allows you to experiment with the following parameters in the `__main__` block:
- **Input Size**: Number of input features (default: `2` for XOR).
- **Hidden Size**: Number of neurons in the hidden layer (default: `2`).
- **Output Size**: Number of output values (default: `1` for XOR).
- **Learning Rate**: Step size for weight updates (default: `0.1`).
- **Epochs**: Number of iterations for training (default: `1000`).

---

## 📈 Example Output

After completing the backpropagation implementation and running the training process, the network should produce outputs close to the target values for the XOR function:
```plaintext
Testing...
Input: [0, 0] - Predicted: [0.05] - Target: [0]
Input: [0, 1] - Predicted: [0.95] - Target: [1]
Input: [1, 0] - Predicted: [0.96] - Target: [1]
Input: [1, 1] - Predicted: [0.04] - Target: [0]
```

---

## 📝 Notes

- This project uses a simple fully connected neural network with one hidden layer.
- All calculations are performed from scratch without external libraries like TensorFlow or PyTorch to provide a clear view of the underlying mechanics.

---

## 🤝 Contributing

Contributions to improve the project or clarify the documentation are welcome! Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

Happy Coding! 🚀
