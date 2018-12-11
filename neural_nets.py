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

        if len(state.size()) < 2:
            state = state.unsqueeze(0)

        # compute hidden with ReLU activation
        hidden = F.relu(self.state2hidden(state))

        # compute output
        action_values = self.hidden2action_values(hidden)

        return action_values


# class ModelNetwork(nn.Module):
#     """
#     Neural network for model of the environment.
#     Input is a state concatenated with the action.
#     Output is:
#         reward
#         next_state
#     """
#
#     def __init__(self, num_hidden=128):
#         nn.Module.__init__(self)
#
#         state_dim = 4
#         action_dim = 2
#
#         self.experience2hidden = nn.Linear(state_dim + action_dim, num_hidden)  # TODO: embedding miss?
#         self.hidden2next_state = nn.Linear(num_hidden, state_dim)
#
#     def forward(self, state_action):
#
#         if len(state_action.size()) < 2:
#             state_action = state_action.unsqueeze(0)
#
#         # compute hidden with ReLU activation
#         hidden = F.relu(self.experience2hidden(state_action))
#
#         # TODO: DO WE WANT BOUNDS FOR THE NEXT STATE?
#         # compute next state
#         next_state = self.hidden2next_state(hidden)
#
#         return next_state


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
        action_dim = 2

        self.experience2hidden = nn.Linear(state_dim + action_dim, num_hidden)  # TODO: embedding miss?
        self.hidden2next_state = nn.Linear(num_hidden, state_dim)

    def forward(self, state_action):

        if len(state_action.size()) < 2:
            state_action = state_action.unsqueeze(0)

        # compute hidden with ReLU activation
        hidden = F.relu(self.experience2hidden(state_action))

        # TODO: DO WE WANT BOUNDS FOR THE NEXT STATE?
        # compute next state
        next_state = torch.tanh(self.hidden2next_state(hidden))
        c_pos, c_vel, p_pos, p_vel = torch.chunk(next_state, 4, dim=-1)
        c_pos = c_pos * 2
        c_vel = c_vel * 3
        p_pos = p_pos * 0.21
        p_vel = p_vel * 3
        next_state = torch.cat([c_pos, c_vel, p_pos, p_vel], dim=-1)

        return next_state