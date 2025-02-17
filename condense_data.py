import torch
import pandas as pd
import pickle

# Load the 3D data from a text file
# Assuming each line in the text file corresponds to a row in the array
data_3d = pd.read_csv('MLINPUT1.txt', delim_whitespace=True, header=None).values

# Convert data to numeric, handling non-numeric values
data_3d = pd.DataFrame(data_3d).apply(pd.to_numeric, errors='coerce').fillna(0).values

# Convert the data to a PyTorch tensor
# Assuming data_3d is already in the correct shape (33, 100*N, 100*N, 100*N, 100*N)
data_tensor = torch.tensor(data_3d, dtype=torch.float32)

# Print the shape of the data_tensor to understand its dimensions
print('Shape of data_tensor:', data_tensor.shape)

# Aggregate the data using mean to condense temperature, pressure, and sigma dependencies
# This will reduce the dimensions to (33, 100)
condensed_tensor = data_tensor.mean(dim=(1, 2, 3, 4))

# Convert the condensed tensor to a Pandas DataFrame
condensed_df = pd.DataFrame(condensed_tensor.numpy(), columns=[f'Radius_{i+1}' for i in range(100)])
condensed_df.insert(0, 'Element', [f'Element_{i+1}' for i in range(33)])

# Save the DataFrame to a CSV file
condensed_df.to_csv('condensed_data.csv', index=False)

# Print the DataFrame to verify
print(condensed_df.head()) 