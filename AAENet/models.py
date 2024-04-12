import torch
import torch.nn as nn

import numpy as np

class Encoder(nn.Module):
    def __init__(self, B=100, R=10):
        # B: number of spectral bands
        # R: size of the abundance vector z
        super(Encoder, self).__init__()

        self.layer_1 = nn.Linear(B, 9*R, bias=False)
        self.layer_2 = nn.Linear(9*R, 6*R, bias=False)
        self.layer_3 = nn.Linear(6*R, 3*R, bias=False)
        self.layer_4 = nn.Linear(3*R, R, bias=False)

        self.batch_norm = nn.BatchNorm1d(R)

        self.threshold = nn.Parameter(2 * (torch.rand(R)-0.5) / np.sqrt(R))

        self.leakyRelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.layer_1(input)
        x = self.leakyRelu(x)

        x = self.layer_2(x)
        x = self.leakyRelu(x)

        x = self.layer_3(x)
        x = self.leakyRelu(x)

        x = self.layer_4(x)
        x = self.leakyRelu(x)

        x = self.batch_norm(x)
        x = self.relu(x)

        # Soft thresholding for the non negativity constraint (ANC)
        x = x - self.threshold
        x = self.relu(x)

        # Normalize to make the sum equal to 1 (ASC)
        x = nn.functional.normalize(x, p=1.0, dim=1)

        return x
    

class Decoder(nn.Module):
    def __init__(self, B=100, R=10, init_endmembers=None, freeze=False):
        # B: number of spectral bands
        # R: size of the abundance vector z
        super(Decoder, self).__init__()

        if init_endmembers is not None:
            self.W = nn.Parameter(init_endmembers, requires_grad=not freeze)
        else:
            self.W = nn.Parameter(torch.abs(torch.rand(R, B)) / np.sqrt(R), requires_grad=not freeze)

    def forward(self, input):
        return input @ self.W
    
    def projection(self):
        # Projected gradient descent in the non-negative orthant 
        with torch.no_grad():
            self.W[self.W<0] = 0


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder  = decoder
    
    def forward(self, input):
        encoding = self.encoder(input)
        return encoding, self.decoder(encoding)
    
    def projection(self):
        self.decoder.projection()


class Discriminator(nn.Module):
    def __init__(self, B=100, R=10):
        # B: number of spectral bands
        # R: size of the abundance vector z
        super(Discriminator, self).__init__()

        n_hidden_1 = max(B // 4, R + 2) + 3
        n_hidden_2 = max(B // 10, R + 1)

        self.layer_1 = nn.Linear(R, n_hidden_1, bias=False)
        self.layer_2 = nn.Linear(n_hidden_1, n_hidden_2, bias=False)
        self.layer_3 = nn.Linear(n_hidden_2, 1, bias=False)

        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.layer_1(input)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.relu(x)

        x = self.layer_3(x)

        return x
    

class AngleDistanceLoss(nn.Module):
    # Spectral Angle Distance (SAD) loss
    def __init__(self):
        super(AngleDistanceLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, x, y, eps=1e-5):
        """
        eps avoids nan values from the arccos because
        numerical values of cosine similarity can sometimes 
        be slightly above 1
        """
        angles = torch.arccos(self.cos_sim(x, y)-eps)
        return angles.mean()
