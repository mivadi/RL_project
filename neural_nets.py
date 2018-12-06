import torch
from torch import nn
import torch.nn.functional as F
# from torch import optim


class QNetwork(nn.Module):

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)

        state_dim = 4
        num_action = 2

        self.state2hidden = nn.Linear(state_dim, num_hidden)
        self.hidden2action_values = nn.Linear(num_hidden, num_action)

    def forward(self, state):

        # compute hidden with ReLU activation
        hidden = F.relu(self.state2hidden(state))

        # compute output
        action_values = self.hidden2action_values(hidden)

        return action_values


class ModelNetwork(nn.Module):
    """
    Neural network for model of the environment.
    Input is a state concatenated with the action.
    Output is:
        reward
        next_state
    """

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)

        state_dim = 4
        action_dim = 1
        reward_dim = 1

        self.experience2hidden = nn.Linear(state_dim + action_dim, num_hidden)
        self.hidden2reward = nn.Linear(num_hidden, reward_dim)
        self.hidden2next_state = nn.Linear(num_hidden, state_dim)

        # self.sigmoid = torch.sigmoid()

    def forward(self, state_action):

        # compute hidden with ReLU activation
        hidden = F.relu(self.experience2hidden(state_action))

        # compute reward
        reward = torch.sigmoid(self.hidden2reward(hidden))

        # Q: DO WE WANT BOUNDS FOR THE NEXT STATE?
        # compute next state
        next_state = self.hidden2next_state(hidden)

        return reward, next_state
