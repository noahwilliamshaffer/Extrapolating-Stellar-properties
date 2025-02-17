import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import pickle

# Load input data from a pickle file
with open('MLINPUT1.pkl', 'rb') as f:
    input_data = pickle.load(f)

# Load output data from a text file
output_data = pd.read_csv('MLOUTPUT1.txt', sep='\s+', header=None).values

# Convert data to PyTorch tensors, handling non-numeric values
input_tensor = torch.tensor(pd.DataFrame(input_data).apply(pd.to_numeric, errors='coerce').fillna(0).values, dtype=torch.float32)
output_tensor = torch.tensor(pd.DataFrame(output_data).apply(pd.to_numeric, errors='coerce').fillna(0).values, dtype=torch.float32)

# Print the shapes of the input and output tensors to verify dimensions
print('Input tensor shape:', input_tensor.shape)
print('Output tensor shape:', output_tensor.shape)

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

# Function to calculate accuracy
def calculate_accuracy(outputs, targets):
    # Calculate the mean absolute error
    mae = torch.mean(torch.abs(outputs - targets))
    # Calculate accuracy as a percentage
    accuracy = 100 * (1 - mae)
    return accuracy

# Modify the training function to continue until 95% accuracy is achieved
def train_until_accuracy(input_file, output_file, model, criterion, optimizer, target_accuracy=95.0):
    # Load input and output data from files
    input_data = pd.read_csv(input_file, delim_whitespace=True, header=None).values
    output_data = pd.read_csv(output_file, delim_whitespace=True, header=None).values

    # Convert data to PyTorch tensors
    input_tensor = torch.tensor(pd.DataFrame(input_data).apply(pd.to_numeric, errors='coerce').fillna(0).values, dtype=torch.float32)
    output_tensor = torch.tensor(pd.DataFrame(output_data).apply(pd.to_numeric, errors='coerce').fillna(0).values, dtype=torch.float32)

    # Create a dataset and data loader
    train_dataset = TensorDataset(input_tensor, output_tensor)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    # Training loop
    epoch = 0
    while True:
        epoch += 1
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            all_outputs = model(input_tensor)
            accuracy = calculate_accuracy(all_outputs, output_tensor)

        print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}%')

        # Check if target accuracy is achieved
        if accuracy >= target_accuracy:
            print(f'Target accuracy of {target_accuracy}% achieved.')
            break

# Initialize the model, loss function, and optimizer
input_size = input_tensor.shape[1]
output_size = output_tensor.shape[1]
model = SimpleNN(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model until 90% accuracy is achieved
train_until_accuracy('MLINPUT1.txt', 'MLOUTPUT1.txt', model, criterion, optimizer, target_accuracy=90.0)

# Save the trained model
torch.save(model.state_dict(), 'model_from_files.pth')

print('Training complete. Model saved as model_from_files.pth.')

# Function to load new data and continue training
def continue_training(new_input_file, new_output_file):
    if os.path.exists('model_from_files.pth'):
        model.load_state_dict(torch.load('model_from_files.pth'))
        print('Model loaded for further training.')
    train_until_accuracy(new_input_file, new_output_file, model, criterion, optimizer)
    torch.save(model.state_dict(), 'model_from_files.pth')
    print('Further training complete. Model updated and saved.') 