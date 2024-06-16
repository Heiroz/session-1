import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1280, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1280, shuffle=False)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # 计算展平后的形状
        conv1_output_size = (28 - 3 + 0) // 1 + 1  # 26x26
        pool1_output_size = conv1_output_size // 2  # 13x13
        conv2_output_size = (pool1_output_size - 3 + 0) // 1 + 1  # 11x11
        pool2_output_size = conv2_output_size // 2  # 5x5
        self.flatten_size = pool2_output_size * pool2_output_size * 64
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

def evaluate(model, device, loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)  # 记录总损失
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    return total_loss, accuracy

for epoch in range(1, num_epochs + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss, train_acc = evaluate(model, device, train_loader)
        test_loss, test_acc = evaluate(model, device, test_loader)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch: {epoch} Batch: {batch_idx} Train Loss: {train_loss:.6f} Train Accuracy: {train_acc:.2f}%')
        print(f'Epoch: {epoch} Batch: {batch_idx} Test Loss: {test_loss:.6f} Test Accuracy: {test_acc:.2f}%')

# 可视化训练和测试表现
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Test loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss per Iteration')
plt.savefig('loss.png')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training accuracy')
plt.plot(test_accuracies, label='Test accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy per Iteration')
plt.savefig('accuracy.png')

plt.show()
