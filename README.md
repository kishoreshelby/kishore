import torch
import torch.nn as nn
import torch.optim as optim


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
