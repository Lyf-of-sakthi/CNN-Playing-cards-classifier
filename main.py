import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
])


card_labels = {
    "ace of hearts": 0, "two of hearts": 1, "three of hearts": 2, "four of hearts": 3, "five of hearts": 4,
    "six of hearts": 5, "seven of hearts": 6, "eight of hearts": 7, "nine of hearts": 8, "ten of hearts": 9,
    "jack of hearts": 10, "queen of hearts": 11, "king of hearts": 12, "ace of spades": 13, "two of spades": 14,
    "three of spades": 15, "four of spades": 16, "five of spades": 17, "six of spades": 18, "seven of spades": 19,
    "eight of spades": 20, "nine of spades": 21, "ten of spades": 22, "jack of spades": 23, "queen of spades": 24,
    "king of spades": 25, "ace of diamonds": 26, "two of diamonds": 27, "three of diamonds": 28, "four of diamonds": 29,
    "five of diamonds": 30, "six of diamonds": 31, "seven of diamonds": 32, "eight of diamonds": 33, "nine of diamonds": 34,
    "ten of diamonds": 35, "jack of diamonds": 36, "queen of diamonds": 37, "king of diamonds": 38, "ace of clubs": 39,
    "two of clubs": 40, "three of clubs": 41, "four of clubs": 42, "five of clubs": 43, "six of clubs": 44,
    "seven of clubs": 45, "eight of clubs": 46, "nine of clubs": 47, "ten of clubs": 48, "jack of clubs": 49,
    "queen of clubs": 50, "king of clubs": 51, "joker": 52
}


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
  
        self.class_to_idx = card_labels  


train_data = CustomImageFolder(root="C:\\Users\\Sakthi\\Downloads\\Cards_dataset\\train", transform=transform)
test_data = CustomImageFolder(root="C:\\Users\\Sakthi\\Downloads\\Cards_dataset\\test", transform=transform)


train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


for images, labels in train_loader:
    print("Labels range:", labels.min().item(), "to", labels.max().item())  
    assert labels.min().item() >= 0 and labels.max().item() < 53, "Labels out of range"
    break


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        with torch.no_grad():  
            dummy_input = torch.randn(1, 3, 128, 128)
            x = self.pool(torch.relu(self.conv1(dummy_input)))
            x = self.pool(torch.relu(self.conv2(x)))
            flattened_size = x.view(1, -1).shape[1]  

        self.fc1 = nn.Linear(flattened_size, 128)  
        self.fc2 = nn.Linear(128, 53) 

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  
        return x


model = SimpleCNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
  
        assert labels.min().item() >= 0 and labels.max().item() < 53, 

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


image_path = "C:\\Users\\Sakthi\\Downloads\\jackofcards.jpg"  
image = Image.open(image_path)


image = transform(image)
image = image.unsqueeze(0).to(device) 


model.eval()
with torch.no_grad():
    output = model(image)
    _, predicted_class = torch.max(output, 1)


class_name = train_data.classes[predicted_class.item()]
print(f"Predicted Class: {class_name}")
