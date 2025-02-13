import pandas as pd
import torch

# Load data from text files into Pandas DataFrames
stellar_df = pd.read_csv('steller_parameters.txt', delim_whitespace=True, header=None)
condensed_dust_df = pd.read_csv('condensed_dust.txt', sep='\t', header=None)

# Convert DataFrames to PyTorch tensors
stellar_df = stellar_df.apply(pd.to_numeric, errors='coerce').fillna(0)
stellar_tensor = torch.tensor(stellar_df.values, dtype=torch.float32)
condensed_dust_tensor = torch.tensor(condensed_dust_df.values, dtype=torch.float32)

# Print the shape of the condensed_dust_tensor to understand its dimensions
print('Shape of condensed_dust_tensor:', condensed_dust_tensor.shape)

# Adjust the aggregation function for a 2D tensor
aggregated_tensor = condensed_dust_tensor.mean(dim=1)

# Convert the aggregated tensor to a Pandas DataFrame with one column per element
aggregated_df = pd.DataFrame(aggregated_tensor.numpy(), columns=['Average_Mass'])
aggregated_df.insert(0, 'Radius', [f'Radius_{i+1}' for i in range(100)])

# Save the DataFrame to a CSV file
aggregated_df.to_csv('aggregated_dust_data.csv', index=False)

# Print the DataFrame to verify
print(aggregated_df.head()) 