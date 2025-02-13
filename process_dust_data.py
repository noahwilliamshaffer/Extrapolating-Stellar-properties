import pandas as pd
import torch

# Load data from text files into Pandas DataFrames
stellar_params_df = pd.read_csv('steller_parameters.txt', header=None)
condensed_dust_df = pd.read_csv('condensed_dust.txt', sep='\t', header=None)

# Convert DataFrames to PyTorch tensors
stellar_params_tensor = torch.tensor(stellar_params_df.values, dtype=torch.float32)
condensed_dust_tensor = torch.tensor(condensed_dust_df.values, dtype=torch.float32)

# Define the number of elements and radius values
num_elements = 33
num_radius = 100

# Reshape the condensed dust tensor to match the desired shape
# Assuming the input shape is (33, 100 * N, 100 * N, 100 * N, 100 * N)
# We need to condense it to (33, 100)
# Here, we use mean to condense temperature, pressure, and sigma dependencies
condensed_tensor = condensed_dust_tensor.view(num_elements, -1, num_radius).mean(dim=1)

# Convert the condensed tensor back to a Pandas DataFrame
condensed_df = pd.DataFrame(condensed_tensor.numpy(), columns=[f'Radius_{i+1}' for i in range(num_radius)])
condensed_df.insert(0, 'Element', [f'Element_{i+1}' for i in range(num_elements)])

# Save the condensed DataFrame to a CSV file
condensed_df.to_csv('condensed_dust_output.csv', index=False)

# Print the condensed DataFrame
print(condensed_df.head()) 