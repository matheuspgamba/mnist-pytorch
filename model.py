import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Camada convolucional: 1 canal de entrada, 32 de saída, kernel 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        # Segunda camada convolucional: 32 de entrada, 64 de saída, kernel 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # Camada totalmente conectada: 9216 entradas (64x12x12) para 128 neurônios
        self.fc1 = nn.Linear(9216, 128)
        # Camada de saída: 128 para 10 classes (dígitos de 0 a 9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)  # Achata o tensor para transformar em um vetor
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)