import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Load input and output data from files
input_data = pd.read_csv('MLINPUT1.txt', delim_whitespace=True, header=None).values
output_data = pd.read_csv('MLOUTPUT1.txt', delim_whitespace=True, header=None).values

# Convert data to PyTorch tensors
input_tensor = torch.tensor(input_data, dtype=torch.float32)
output_tensor = torch.tensor(output_data, dtype=torch.float32)

# Create a dataset and data loader
train_dataset = TensorDataset(input_tensor, output_tensor)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = input_tensor.shape[1]
output_size = output_tensor.shape[1]
model = SimpleNN(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'model_from_files.pth')

print('Training complete. Model saved as model_from_files.pth.') 