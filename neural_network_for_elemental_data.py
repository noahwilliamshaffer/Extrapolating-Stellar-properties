import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Function to read and preprocess input data

def preprocess_input_data(file_path):
    # Read the input file
    try:
        data = pd.read_csv(file_path, sep=r'\s+', header=None)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

    # Inspect and preprocess the data
    # Handle missing values, normalize, and reshape as needed
    # For now, assume data is correctly formatted
    # Coerce non-numeric values to numeric and fill NaNs with zeros
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    data_tensor = torch.tensor(data.values, dtype=torch.float32)

    return data_tensor

# Load and preprocess input data
input_tensor = preprocess_input_data('inputfile.txt')

# Placeholder for output data loading
# output_tensor = preprocess_input_data('outputfile.txt')

# Check if input data is loaded
if input_tensor is not None:
    print('Input data loaded successfully.')
else:
    print('Failed to load input data.')

# Print the shapes of the input and output tensors to debug size mismatch
print('Input tensor shape:', input_tensor.shape)

# Load and preprocess output data
output_tensor = preprocess_input_data('outputfile.txt')

# Check if output data is loaded
if output_tensor is not None:
    print('Output data loaded successfully.')
else:
    print('Failed to load output data.')

# Print the shapes of the input and output tensors to debug size mismatch
print('Output tensor shape:', output_tensor.shape)

# Create a dataset and data loader
train_dataset = TensorDataset(input_tensor, output_tensor)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

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

# Evaluate the model
with torch.no_grad():
    predictions = model(input_tensor)
    comparison_result = torch.allclose(predictions, output_tensor, atol=1e-2)

# Print comparison result
print('Model predictions are close to the output data:', comparison_result)

# Further steps will include model definition, training, and evaluation
# This is just the initial setup for data loading and preprocessing. 