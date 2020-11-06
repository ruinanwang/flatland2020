import torch.nn as nn
import torch.nn.functional as F
import torch


class DuelingQNetwork(nn.Module):
    """Dueling Q-network (https://arxiv.org/abs/1511.06581)"""

    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(DuelingQNetwork, self).__init__()

        # value network
        self.cnn1_val = nn.Conv2d(in_channels=7, out_channels=14, kernel_size=3, stride=3, padding=2)
        nn.init.kaiming_uniform_(self.cnn1_val.weight)
        self.pooling1_val = nn.MaxPool2d(3,3)
        self.cnn2_val = nn.Conv2d(in_channels=14, out_channels=28, kernel_size=3, stride=3, padding=2)
        nn.init.kaiming_uniform_(self.cnn2_val.weight)
        self.cnn3_val = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=3, stride=3, padding=2)
        self.fc2_val = nn.Linear(224, hidsize2)
        self.fc3_val = nn.Linear(hidsize2, 50)
        self.fc4_val = nn.Linear(50, 1)

        # advantage network
        self.cnn1_adv = nn.Conv2d(in_channels=7, out_channels=14, kernel_size=3, stride=3, padding=2)
        nn.init.kaiming_uniform_(self.cnn1_adv.weight)
        self.pooling1_adv = nn.MaxPool2d(3,3)
        self.cnn2_adv = nn.Conv2d(in_channels=14, out_channels=28, kernel_size=3, stride=3, padding=2)
        nn.init.kaiming_uniform_(self.cnn2_adv.weight)
        self.cnn3_adv = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=3, stride=3, padding=2)
        self.fc2_adv = nn.Linear(224, hidsize2)
        self.fc3_adv = nn.Linear(hidsize2, 50)
        self.fc4_adv = nn.Linear(50, action_size)

    def forward(self, x):
        x = torch.reshape(x, (-1,7,25,25))
        x = x.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        val = self.pooling1_val(F.relu(self.cnn1_val(x)))
        val = F.relu(self.cnn2_val(val))
        val = F.relu(self.cnn3_val(val))
        val = nn.Flatten()(val)
        val = F.relu(self.fc2_val(val))
        val = F.relu(self.fc3_val(val))
        val = self.fc4_val(val)

        # advantage calculation
        adv = self.pooling1_adv(F.relu(self.cnn1_adv(x)))
        adv = F.relu(self.cnn2_adv(adv))
        adv = F.relu(self.cnn3_adv(adv))
        adv = nn.Flatten()(adv)
        adv = F.relu(self.fc2_adv(adv))
        adv = F.relu(self.fc3_adv(adv))
        adv = self.fc4_adv(adv)

        return val + adv - adv.mean()
