import torch.nn as nn
import torch.nn.functional as F
import torch


class DuelingQNetwork(nn.Module):
    """Dueling Q-network (https://arxiv.org/abs/1511.06581)"""

    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(DuelingQNetwork, self).__init__()

        # value network
        self.cnn_val = nn.Conv2d(in_channels=7, out_channels=14, kernel_size=3, stride=3, padding=2)
        self.fc2_val = nn.Linear(1134, hidsize2)
        self.fc4_val = nn.Linear(hidsize2, 1)

        # advantage network
        self.cnn_adv = nn.Conv2d(in_channels=7, out_channels=14, kernel_size=3, stride=3, padding=2)
        self.fc2_adv = nn.Linear(1134, hidsize2)
        self.fc4_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        x = torch.reshape(x, (-1,7,25,25))
        val = F.relu(self.cnn_val(x))
        val = nn.Flatten()(val)
        val = F.relu(self.fc2_val(val))
        val = self.fc4_val(val)

        # advantage calculation
        adv = F.relu(self.cnn_adv(x))
        adv = nn.Flatten()(adv)
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc4_adv(adv)

        return val + adv - adv.mean()
