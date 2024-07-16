import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=1)
        self.fc1 = nn.Linear(3, 64)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=1)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 9)  # Assuming 9 classes for classification
        
    def global_max_pooling(self, x):
        x, _ = torch.max(x, dim=1)  # Apply max pooling across the num_channels dimension
        return x  # Returns a tensor of shape [batch_size, num_features]

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        
        # Initialize a list to hold the outputs of fc1
        fc1_outputs = []
        
        print("after fc1", x.shape)
        
        # Loop through each channel
        for i in range(x.size(1)):
            channel_data = x[:, i, :]  # Select the data for the i-th channel
            fc_output = self.fc1(channel_data)
            fc1_outputs.append(fc_output.unsqueeze(1)) 
        
        # Combine the outputs along the new dimension
        x = torch.cat(fc1_outputs, dim=1)
        
        print("after fc1", x.shape)        
        
        fc2_outputs = []
        
        for i in range(x.size(1)):
            channel_data = x[:, i, :]  # Select the data for the i-th channel
            fc_output = self.fc2(channel_data)
            fc2_outputs.append(fc_output.unsqueeze(1))  # Unsqueeze to add a new dimension
        
        x = torch.cat(fc2_outputs, dim=1)
        
        print("after fc2", x.shape)
        
        x = self.conv2(x)
        x = torch.relu(x)
        
        print("after conv2", x.shape)
        
        fc3_outputs = []
        
        # Loop through each channel
        for i in range(x.size(1)):
            channel_data = x[:, i, :]  # Select the data for the i-th channel
            fc_output = self.fc3(channel_data)
            fc3_outputs.append(fc_output.unsqueeze(1)) 
        
        # Combine the outputs along the new dimension
        x = torch.cat(fc3_outputs, dim=1)
        
        print("after fc3", x.shape)
        
        fc4_outputs = []
        
        for i in range(x.size(1)):
            channel_data = x[:, i, :]  # Select the data for the i-th channel
            fc_output = self.fc4(channel_data)
            fc4_outputs.append(fc_output.unsqueeze(1))  # Unsqueeze to add a new dimension
        
        x = torch.cat(fc4_outputs, dim=1)
        
        print("after fc4", x.shape)
        
        x = self.global_max_pooling(x)
        
        print("after pooling", x.shape)
        
        x = self.fc5(x)
        x = torch.relu(x)
        
        print("after fc5", x.shape)
        
        x = self.fc6(x)
        x = torch.relu(x)
        
        print("after fc6", x.shape)
        
        x = self.fc7(x)
        
        print("after fc7", x.shape)
        
        return x  # Output logits for classification

# Function to generate random labels
def generate_random_labels(batch_size):
    return torch.randint(0, 9, (batch_size,), dtype=torch.long)

# Function to calculate accuracy
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

import pandas as pd
import numpy as np
import torch

# Load the CSV file
df = pd.read_csv('output_data.csv')

# Drop unnecessary columns
df = df.drop(columns=['S.no', 'R', 'G', 'B'])

# Define X (features) and Y (labels)
X = df[['X', 'Y', 'Z']]
Y = df['Class/Label']

# Convert X and Y to NumPy arrays
X_train = X.values
Y_train_orig = Y.values  # Original Y values

# Example X_train shape: (4723353, 3)

# Create an empty tensor to store the final transformed data
transformed_data = torch.zeros(100, 20, 3)

# Iterate 10 times to randomly pick 20 points from X_train each time
for i in range(10):
    # Randomly select 20 indices
    random_indices = np.random.choice(len(X_train), size=20, replace=False)
    
    # Extract the selected points
    selected_points = X_train[random_indices]
    
    # Assign the selected points to the appropriate position in the transformed_data tensor
    transformed_data[i*10:(i+1)*10, :, :] = torch.tensor(selected_points, dtype=torch.float32)

# Create Y_train from Y_train_orig
Y_train = torch.zeros(100, dtype=torch.long)  # Initialize Y_train tensor

# Iterate over transformed_data and compute the average label for each batch of 20 points
for i in range(10):
    # Get the labels corresponding to the selected points
    labels_batch = Y_train_orig[random_indices]
    
    # Calculate the average label for this batch
    average_label = torch.tensor(np.mean(labels_batch), dtype=torch.long)
    
    # Assign this average label to the corresponding 20 points
    Y_train[i*10:(i+1)*10] = average_label

# Reshape Y_train to (10, 10)
Y_train = Y_train.view(10, 10)

# Print the shape of transformed_data and Y_train
print("Transformed Data shape:", transformed_data.shape)
print("Y_train shape:", Y_train.shape)



# Create the model
model = NN()

# Set up the optimizer (Adam) and loss function (CrossEntropyLoss)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Number of training epochs and batch size
num_epochs = 10
batch_size = 10

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    for i in range(0, len(transformed_data), batch_size):
        # Get the batch inputs and labels
        inputs = transformed_data[i:i+batch_size]
        targets = Y_train[epoch]
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate loss for the epoch
        epoch_loss += loss.item()
        
        # Calculate batch accuracy
        batch_accuracy = calculate_accuracy(outputs, targets)
        
        # Accumulate accuracy for the epoch
        epoch_accuracy += batch_accuracy
        
        # Print statistics
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(transformed_data)}], Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}')
    
    # Average epoch loss and accuracy
    epoch_loss /= (len(transformed_data) / batch_size)
    epoch_accuracy /= (len(transformed_data) / batch_size)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Print the shape of the new combined output (not needed for training, just for information)
print(f'Training completed. Final output shape: {outputs.shape}')
