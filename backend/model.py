import torch.nn as nn

class InjuryClassifierCNN(nn.Module):
    def __init__(self):
        super(InjuryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 64 * 64, 6)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
