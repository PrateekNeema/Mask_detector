import os
import torch
from torch import nn

class NeuralNetwork(nn.Module):

    def __init__(self):

        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()

        self.first_stack = nn.Sequential(
            nn.Linear(400*400*3, 512),      #assuming images are of 400*400 dimensions????#please verify this
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):     #where x is the training image(can be in 400,400,3 form as we are flattening it anyways

        x = self.flatten(x)
        logits = self.first_stack(x)
        return logits           # returning predicted value
