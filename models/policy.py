import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn.utils as utils
import math


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class Policy_CNN(nn.Module):
    def __init__(self, hidden_dim=10, actions=7, num_layers=19, in_channels=64):
        super(Policy_CNN, self).__init__()
        self.actions = actions
        self.num_layers = num_layers

        self.reduce = nn.Sequential(
            conv_dw(in_channels, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, actions * num_layers * 2)
        )

    def forward(self, x):
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.view(-1, 2, self.num_layers, self.actions)        # 2 - Activations and weights
        x_a, x_w = x.split(split_size=1, dim=1)
        probs_a = F.softmax(x_a.squeeze(), dim=2)
        probs_w = F.softmax(x_w.squeeze(), dim=2)
        m_a = Categorical(probs_a)
        m_w = Categorical(probs_w)
        bit_a = m_a.sample()
        bit_w = m_w.sample()

        return (bit_a + 2, bit_w + 2), (m_a.log_prob(bit_a), m_w.log_prob(bit_w))


# Predict one layer at a time, taking into account the previous layers
class Policy_RNN(nn.Module):
    def __init__(self, hidden_dim=10, actions=7, num_layers=19, in_channels=64):
        super(Policy_RNN, self).__init__()
        self.actions = actions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_layers = 2

        self.LSTM = nn.LSTM(hidden_dim + 2, hidden_dim, self.lstm_layers)
        self.hidden2logits = nn.Linear(hidden_dim, actions * 2)

        self.reduce = nn.Sequential(
            conv_dw(in_channels, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def init_hidden(self, batch_size):
        return (torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, requires_grad=True).cuda(),
                torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, requires_grad=True).cuda())

    def forward(self, x):
        batch_size = x.size(0)

        # x = x.detach()
        x = self.reduce(x)
        x = torch.flatten(x, 1)

        # Init hidden state vectors with zeros
        hidden = self.init_hidden(batch_size)
        self.LSTM.flatten_parameters()

        # Initialize bitwidth vectors and log_probs
        bit_a = torch.empty(0, dtype=torch.int64, device=x.device)
        bit_w = torch.empty(0, dtype=torch.int64, device=x.device)
        log_probs_a = torch.empty(0, dtype=torch.float32, device=x.device)
        log_probs_w = torch.empty(0, dtype=torch.float32, device=x.device)

        # Pad input with zeros (zeros will be filled with the sequentially predicted bitwidths)
        zeros = torch.zeros((batch_size, 2), dtype=x.dtype, device=x.device)
        inp = []
        inp.append(torch.cat((x, zeros), dim=1))         # inp = [x, 0, 0]

        # Loop through layers
        for i in range(0, self.num_layers):
            out, hidden = self.LSTM(inp[i].view(1, batch_size, -1), hidden)      # Sequence is length 1
            out = out.squeeze()
            logits = self.hidden2logits(out)
            logits_a, logits_w = logits.split(split_size=self.actions, dim=1)
            probs_a = F.softmax(logits_a, dim=1)
            probs_w = F.softmax(logits_w, dim=1)
            m_a = Categorical(probs_a)
            m_w = Categorical(probs_w)
            _bit_a = m_a.sample()
            _bit_w = m_w.sample()

            # Append bits and log_probs
            bit_a = torch.cat((bit_a, _bit_a.unsqueeze(0)), dim=0)
            bit_w = torch.cat((bit_w, _bit_w.unsqueeze(0)), dim=0)
            log_probs_a = torch.cat((log_probs_a, m_a.log_prob(_bit_a).unsqueeze(0)), dim=0)
            log_probs_w = torch.cat((log_probs_w, m_w.log_prob(_bit_w).unsqueeze(0)), dim=0)

            # Next input
            subseq = torch.zeros((batch_size, 2), dtype=x.dtype, device=x.device)
            subseq[:, 0] = _bit_a
            subseq[:, 1] = _bit_w
            inp.append(torch.cat((x, subseq), dim=1))

        # Transpose bits and log_probs to correct shape
        bit_a = bit_a.t()
        bit_w = bit_w.t()
        log_probs_a = log_probs_a.t()
        log_probs_w = log_probs_w.t()

        return (bit_a + 2, bit_w + 2), (log_probs_a, log_probs_w)


pi = torch.FloatTensor([math.pi]).cuda()


def normal(x, mu, sigma_sq):
    a = (-1*(x-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b


class Policy_CNN_continous(nn.Module):
    def __init__(self, hidden_dim=10, num_layers=19, in_channels=64):
        super(Policy_CNN_continous, self).__init__()
        self.num_layers = num_layers

        self.reduce = nn.Sequential(
            conv_dw(in_channels, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_layers * 4)    # mu_a, mu_w, var_a, var_w
        )

    def discretize(self, x):
        x[x < 0] = 0
        x[x > 7] = 7
        x = x.floor().long()
        return x

    def forward(self, x):
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.view(-1, 4, self.num_layers)    # mu_a, mu_w, var_a, var_w

        mu = F.relu(x[:, 0:2]+1)
        var = F.softplus(x[:, 2:4])
        mu_a, mu_w = mu[:, 0], mu[:, 1]
        var_a, var_w = var[:, 0], var[:, 1]

        eps_a = torch.randn(mu_a.size(), dtype=mu_a.dtype, device=mu_a.device)
        eps_w = torch.randn(mu_w.size(), dtype=mu_w.dtype, device=mu_w.device)

        # Calculate the probability
        action_a = (mu_a + var_a.sqrt() * eps_a).data
        action_w = (mu_w + var_w.sqrt() * eps_w).data
        probs_a = normal(action_a, mu_a, var_a)
        probs_w = normal(action_w, mu_w, var_w)

        bit_a = self.discretize(action_a)
        bit_w = self.discretize(action_w)

        return (bit_a + 2, bit_w + 2), (probs_a.log(), probs_w.log())


class Policy_RNN_continous(nn.Module):
    def __init__(self, hidden_dim=10, num_layers=19, in_channels=64):
        super(Policy_RNN_continous, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_layers = 2

        # self.LSTM = nn.LSTM(hidden_dim + 2, hidden_dim, self.lstm_layers)
        self.LSTM = nn.LSTM(hidden_dim + 2, hidden_dim, self.lstm_layers)
        self.hidden2mu = nn.Linear(hidden_dim, 1*2)
        self.hidden2var = nn.Linear(hidden_dim, 1*2)
        self.regressor = nn.Linear(1024, hidden_dim)

        self.reduce = nn.Sequential(
            conv_dw(in_channels, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def init_hidden(self, batch_size):
        return (torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, requires_grad=True).cuda(),
                torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, requires_grad=True).cuda())

    def discretize(self, x):
        x[x < 0] = 0
        x[x > 6] = 6
        x = x.round().long()
        return x

    def forward(self, x):
        batch_size = x.size(0)

        x = self.reduce(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)

        # Init hidden state vectors with zeros
        hidden = self.init_hidden(batch_size)
        self.LSTM.flatten_parameters()

        # Initialize bitwidth vectors and log_probs
        bit_a = torch.empty(0, dtype=torch.int64, device=x.device)
        bit_w = torch.empty(0, dtype=torch.int64, device=x.device)
        log_probs_a = torch.empty(0, dtype=torch.float32, device=x.device)
        log_probs_w = torch.empty(0, dtype=torch.float32, device=x.device)
        entropies_a = torch.empty(0, dtype=torch.float32, device=x.device)
        entropies_w = torch.empty(0, dtype=torch.float32, device=x.device)

        # Pad input with zeros (zeros will be filled with the sequentially predicted bitwidths)
        zeros = torch.zeros((batch_size, 2), dtype=x.dtype, device=x.device)
        inp = []
        inp.append(torch.cat((x, zeros), dim=1))         # inp = [x, 0, 0]

        # Loop through layers
        for i in range(0, self.num_layers):
            out, hidden = self.LSTM(inp[i].view(1, batch_size, -1), hidden)      # Sequence is length 1
            out = out.squeeze()
            mu = F.softplus(self.hidden2mu(out))
            var = F.softplus(self.hidden2var(out))
            mu_a, mu_w = mu[:, 0], mu[:, 1]
            var_a, var_w = var[:, 0], var[:, 1]

            eps_a = torch.randn(mu_a.size(), dtype=mu_a.dtype, device=mu_a.device)
            eps_w = torch.randn(mu_w.size(), dtype=mu_w.dtype, device=mu_w.device)

            # Calculate the probability
            action_a = (mu_a + var_a.sqrt() * eps_a).data
            action_w = (mu_w + var_w.sqrt() * eps_w).data
            prob_a = normal(action_a, mu_a, var_a)
            prob_w = normal(action_w, mu_w, var_w)
            entropy_a = -0.5 * ((var_a + 2 * pi.expand_as(var_a)).log() + 1)
            entropy_w = -0.5 * ((var_w + 2 * pi.expand_as(var_w)).log() + 1)

            _bit_a = self.discretize(action_a)
            _bit_w = self.discretize(action_w)

            # Append bits and log_probs
            bit_a = torch.cat((bit_a, _bit_a.unsqueeze(0)), dim=0)
            bit_w = torch.cat((bit_w, _bit_w.unsqueeze(0)), dim=0)
            log_probs_a = torch.cat((log_probs_a, prob_a.log().unsqueeze(0)), dim=0)
            log_probs_w = torch.cat((log_probs_w, prob_w.log().unsqueeze(0)), dim=0)
            entropies_a = torch.cat((entropies_a, entropy_a.unsqueeze(0)), dim=0)
            entropies_w = torch.cat((entropies_w, entropy_w.unsqueeze(0)), dim=0)

            # Next input
            subseq = torch.zeros((batch_size, 2), dtype=x.dtype, device=x.device)
            subseq[:, 0] = _bit_a
            subseq[:, 1] = _bit_w
            inp.append(torch.cat((x, subseq), dim=1))


        # Transpose bits and log_probs to correct shape
        bit_a = bit_a.t()
        bit_w = bit_w.t()
        log_probs_a = log_probs_a.t()
        log_probs_w = log_probs_w.t()
        entropies_a = entropies_a.t()
        entropies_w = entropies_w.t()

        return (bit_a + 2, bit_w + 2), (log_probs_a, log_probs_w), (entropies_a, entropies_w)


class Policy_RNN_continous_layerwise(nn.Module):
    def __init__(self, hidden_dim=10, num_layers=19, in_channels=[]):
        super(Policy_RNN_continous_layerwise, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_layers = 2

        self.LSTM = nn.LSTM(hidden_dim, hidden_dim, self.lstm_layers)
        self.hidden2mu = nn.Linear(hidden_dim, 1)
        self.hidden2var = nn.Linear(hidden_dim, 1)

        self.layer = 0
        reduce = []
        for c in in_channels:
            reduce.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(3),
                nn.Conv2d(in_channels=c, out_channels=hidden_dim, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(hidden_dim),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)
                )
            )
        self.reduce = nn.ModuleList(reduce)

        # Initialize bitwidth vectors and log_probs
        self.bit_a = torch.empty(0, dtype=torch.int64).cuda()
        self.log_probs_a = torch.empty(0, dtype=torch.float32).cuda()
        self.entropies_a = torch.empty(0, dtype=torch.float32).cuda()
        self.bit_w = torch.empty(0, dtype=torch.int64).cuda()
        self.log_probs_w = torch.empty(0, dtype=torch.float32).cuda()
        self.entropies_w = torch.empty(0, dtype=torch.float32).cuda()

        self.hidden = None

    def init_hidden(self, batch_size):
        return (torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, requires_grad=True).cuda(),
                torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, requires_grad=True).cuda())

    def init_vectors(self):
        self.bit_a = torch.empty(0, dtype=torch.int64).cuda()
        self.log_probs_a = torch.empty(0, dtype=torch.float32).cuda()
        self.entropies_a = torch.empty(0, dtype=torch.float32).cuda()
        self.bit_w = torch.empty(0, dtype=torch.int64).cuda()
        self.log_probs_w = torch.empty(0, dtype=torch.float32).cuda()
        self.entropies_w = torch.empty(0, dtype=torch.float32).cuda()

    def discretize(self, x):
        x[x < 0] = 0
        x[x > 6] = 6
        x = x.round().long()
        return x

    def forward(self, input):
        x, downsample, eightbit = input
        batch_size = x.size(0)

        x = self.reduce[self.layer](x)
        x = x.squeeze()

        if self.layer == 0:
            # Init hidden state vectors with zeros
            self.hidden = self.init_hidden(batch_size)
            self.init_vectors()
            self.LSTM.flatten_parameters()

        # Loop through layers
        out, self.hidden = self.LSTM(x.view(1, batch_size, -1), self.hidden)      # Sequence is length 1
        out = out.squeeze()
        mu_a = F.softplus(self.hidden2mu(out)).squeeze()
        var_a = F.softplus(self.hidden2var(out)).squeeze()

        eps_a = torch.randn(mu_a.size(), dtype=mu_a.dtype, device=mu_a.device)

        # Calculate the probability
        action_a = (mu_a + var_a.sqrt() * eps_a).data
        prob_a = normal(action_a, mu_a, var_a)
        entropy_a = -0.5 * ((var_a + 2 * pi.expand_as(var_a)).log() + 1)

        _bit_a = self.discretize(action_a)
        if eightbit:
            _bit_a[:] = 1
        _bit_w = torch.zeros_like(_bit_a)


        # Append bits and log_probs
        self.bit_a = torch.cat((self.bit_a, _bit_a.unsqueeze(0)), dim=0)
        self.log_probs_a = torch.cat((self.log_probs_a, prob_a.log().unsqueeze(0)), dim=0)
        self.entropies_a = torch.cat((self.entropies_a, entropy_a.unsqueeze(0)), dim=0)
        self.bit_w = torch.cat((self.bit_w, _bit_w.unsqueeze(0)), dim=0)
        self.log_probs_w = self.log_probs_a.clone()
        self.entropies_w = self.entropies_a.clone()

        # Increase layer counter
        self.layer += 1

        return (_bit_a + 2, _bit_w + 2)

    def get_predictions(self):
        self.layer = 0

        # Transpose bits and log_probs to correct shape
        return (self.bit_a.t() + 2, self.bit_w.t() + 2), \
               (self.log_probs_a.t(), self.log_probs_w.t()), \
               (self.entropies_a.t(), self.entropies_w.t())

    def configure_layer(self, conv, bn, bit):
        conv.bitA = bit[0]
        conv.bitW = bit[1]
        bn.switch = bit[0] - 2         # Switches range is 0-6, bitwidths range is 2-8


class Policy_CNN_continous_whole(nn.Module):
    def __init__(self, hidden_dim=10, num_layers=19, in_channels=64):
        super(Policy_CNN_continous_whole, self).__init__()
        self.num_layers = num_layers

        self.reduce = nn.Sequential(
            conv_dw(in_channels, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            conv_dw(hidden_dim, hidden_dim, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.hidden2mu = nn.Linear(hidden_dim, 2)         # mu_a, mu_w
        self.hidden2var = nn.Linear(hidden_dim, 2)        # var_a, var_w

    def discretize(self, x):
        x[x < 0] = 0
        x[x > 6] = 6
        x = x.round().long()
        return x

    def forward(self, x):
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        mu = F.softplus(self.hidden2mu(x))
        var = F.softplus(self.hidden2var(x))

        mu_a, mu_w = mu[:, 0], mu[:, 1]
        var_a, var_w = var[:, 0], var[:, 1]

        eps_a = torch.randn(mu_a.size(), dtype=mu_a.dtype, device=mu_a.device)
        eps_w = torch.randn(mu_w.size(), dtype=mu_w.dtype, device=mu_w.device)

        # Calculate the probability
        action_a = (mu_a + var_a.sqrt() * eps_a).data
        action_w = (mu_w + var_w.sqrt() * eps_w).data
        probs_a = normal(action_a, mu_a, var_a)
        probs_w = normal(action_w, mu_w, var_w)
        entropy_a = -0.5 * ((var_a + 2 * pi.expand_as(var_a)).log() + 1)
        entropy_w = -0.5 * ((var_w + 2 * pi.expand_as(var_w)).log() + 1)

        bit_a = self.discretize(action_a)
        bit_w = self.discretize(action_w)
        bit_a = bit_a.unsqueeze(-1).repeat(1, self.num_layers)
        bit_w = bit_w.unsqueeze(-1).repeat(1, self.num_layers)

        return (bit_a + 2, bit_w + 2), (probs_a.log(), probs_w.log()), (entropy_a, entropy_w)
