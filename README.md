# XOR Gate Prediction using Artificial Neural Network

This project demonstrates how a simple artificial neural network can learn the XOR logic function using PyTorch.

The XOR gate is a classic example in machine learning because it is not linearly separable. A single perceptron cannot solve it, but a neural network with at least one hidden layer can.

## Project Objective

Train a neural network to predict the XOR output for the following inputs:

- [0, 0] -> 0
- [0, 1] -> 1
- [1, 0] -> 1
- [1, 1] -> 0

## Requirements

- Python 3.x
- PyTorch

Install dependencies:

```bash
pip install torch
```

## Project Files

- `xor_ann.py` contains the complete PyTorch implementation.

## Model Overview

The neural network includes:
- Input layer with 2 neurons
- Hidden layer with multiple neurons
- Output layer with 1 neuron
- Activation function: GELU / ReLU / Sigmoid
- Loss function: Mean Squared Error (MSE)

## Code

```python
import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], dtype=torch.float32)

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.GELU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

num_epochs = 5000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'epoch: {epoch}, loss={loss.item():.4f}, prediction={y_pred.detach().numpy().ravel()}')

with torch.no_grad():
    print("Final predictions:")
    print(model(X))
    print("Rounded outputs:")
    print(torch.round(model(X)).numpy())
```

## Expected Output

After training, the model should predict approximately:

- [0, 0] -> 0
- [0, 1] -> 1
- [1, 0] -> 1
- [1, 1] -> 0

## Why XOR Is Important

The XOR problem is a foundational example in neural networks because it shows:
- why non-linear activation functions matter
- why hidden layers are necessary
- how backpropagation helps a network learn complex patterns

## Notes

This project is intended as a beginner-friendly introduction to neural networks and deep learning. It is a simple but effective example of how a model learns logical patterns from data.

## License

This project is provided for educational purposes.
