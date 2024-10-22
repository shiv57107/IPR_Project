import pickle

file_path = '/scratch/pranjal/DEN/models/den_gen2_v122orig/val_loss.pkl'

# Open the pickle file in binary read mode
with open(file_path, 'rb') as file:
    # Load the pickle data
    data = pickle.load(file)

# Print the loaded data
print(data)