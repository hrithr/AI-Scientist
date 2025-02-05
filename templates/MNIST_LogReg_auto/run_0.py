import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
import os
import json
import numpy as np

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="run_0")
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784  # 28x28 images flattened
num_classes = 10
learning_rate = 0.1
batch_size = 64
num_epochs = 1

# MNIST dataset
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)

model = LogisticRegression().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
all_results = {}
final_infos = {}

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            predictions = scores.argmax(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

        accuracy = float(num_correct) / float(num_samples)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        all_results[f'Epoch_{epoch}'] = {"Accuracy": [accuracy]}
        final_infos[f'Epoch_{epoch}'] = {"means": {"Accuracy_mean": accuracy}}

# Save the results
os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
    np.save(f, all_results)
with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
    json.dump(final_infos, f)