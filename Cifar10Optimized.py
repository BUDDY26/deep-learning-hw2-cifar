import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Enable CUDA Benchmarking for Performance Optimization
torch.backends.cudnn.benchmark = True

def get_dataloaders(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return trainloader, valloader, testloader

class MLP(nn.Module):
    def __init__(self, input_size=3072, hidden_sizes=[256, 128, 64], num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
        
        self.dropout1 = nn.Dropout(0.3)  # Lower dropout for first layers
        self.dropout2 = nn.Dropout(0.5)  # Higher dropout for deeper layers
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout2(x)  # Stronger dropout before final layer
        x = self.fc4(x)
        if not self.training:
            x = self.softmax(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainloader, valloader, testloader = get_dataloaders(batch_size=256)

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.amp.GradScaler('cuda')

epochs = 30
train_loss, train_acc, val_loss, val_acc = [], [], [], []

for epoch in range(epochs):
    model.train()
    correct, total, running_loss = 0, 0, 0.0
    for images, labels in trainloader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    train_loss.append(running_loss / len(trainloader))
    train_acc.append(100 * correct / total)
    
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    val_loss.append(running_loss / len(valloader))
    val_acc.append(100 * correct / total)
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.2f}%, Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_acc[-1]:.2f}%')

model.eval()
correct, total, test_loss = 0, 0, 0.0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

final_accuracy = 100 * correct / total
print(f'Test Accuracy: {final_accuracy:.2f}%, Test Loss: {test_loss:.4f}')

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_loss, label='Train Loss', color='blue')
plt.plot(range(1, epochs+1), val_loss, label='Val Loss', linestyle='dashed', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), train_acc, label='Train Accuracy', color='blue')
plt.plot(range(1, epochs+1), val_acc, label='Val Accuracy', linestyle='dashed', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)

plt.show()
