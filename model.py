import numpy as np
import matplotlib.pyplot as plt
from discrete_states import compute_bins, converge_state
from tqdm import tqdm as _tqdm
from windy_gridworld import WindyGridworldEnv
from gridworld import GridworldEnv
from collections import defaultdict
from helpers import make_epsilon_greedy_policy, q_learning, is_done
import random
from abc import abstractmethod
import gym

import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from neural_nets import QNetwork, ModelNetwork


class DynaQ(object):
    """
    DynaQ base class.
    """

    def __init__(self, env, planning_steps=1, discount_factor=1., lr=0.5, epsilon=0.1):

        # set environment
        self.environment = env

        # only once load the bins and values for making the continuos values discrete
        # compute_bins(c_pos_bounds, c_vel_bounds, p_pos_bounds, p_vel_bounds, n_bins=10):
        self.edges, self.averages = compute_bins([-2.4, 2.4], [-1.5, 1.5], [-0.21, 0.21], [-1.5, 1.5])

        # set parameters
        self._discount_factor = discount_factor
        self._learning_rate = lr
        self._epsilon = epsilon
        self._planning_steps = planning_steps

        # keep track of already visited state-action pairs
        self.visited_pairs = {}
        self.pair_count = 0

        # initialize Q
        self.Q = None
        self.edges = None
        self.averages = None

        # initialize stats
        self.episode_lengths = []
        self.total_rewards = []

    def reset_data(self):
        self.episode_lengths = []
        self.total_rewards = []

    def set_discount_factor(self, gamma):

        self._discount_factor = gamma

    def set_learning_rate(self, lr):

        self._learning_rate = lr

    def set_epsilon(self, eps):

        self._epsilon = eps

    def set_planning_steps(self, n):

        self._planning_steps = n

    def add_state_action_pair(self, state, action):
        """
        Keep track of already visited state-action pairs for planning.
        """
        if state not in self.visited_pairs:
            self.visited_pairs[state] = set()
        if action not in self.visited_pairs[state]:
            self.pair_count += 1
        self.visited_pairs[state].add(action)

    def policy_fn(self, state):

        # get random number
        random_number = random.uniform(0, 1)

        # get actions with maximum value
        greedy_actions = np.argwhere(self.action_values(state) == np.amax(self.action_values(state))).squeeze()
        if not len(greedy_actions.shape):
            greedy_actions = [greedy_actions]
        action = random.choice(greedy_actions)

        # if number less than epsilon, get random other actions
        if random_number < self._epsilon:
            all_actions = list(range(0, self.environment.action_space.n))
            if not len(greedy_actions) == self.environment.action_space.n:
                action = random.choice(all_actions)

        return int(action)

    def q_learning(self, state):
        """
        Tabular one-step Q-learning algorithm. Takes an action according to a current state and updates the action-value
        function accordingly.


        :param policy: policy function that takes state and returns action
        :param state: OpenAI state
        :return: Tuple of next state, observerd reward and done bool
        """

        # choose A from S using policy derived from Q (epsilon-greedy)
        action = self.policy_fn(state)

        # take action, observe R, S'
        (next_state, reward, done, probability) = self.environment.step(int(action))

        next_state = tuple(converge_state(next_state, self.edges, self.averages))

        self.update_action_value_function(state, next_state, action, reward, done)

        # copy state_tilde and action_tilde to be next steps.
        state = next_state

        return action, state, reward, done

    def planning(self):
        """
        Plan n steps ahead based on already seen state-action pairs and update Q accordingly
        """

        for i in range(self._planning_steps):

            state = random.choice(list(self.visited_pairs.keys()))
            action = random.choice(list(self.visited_pairs[state]))
            next_state, reward = self.model(state, action)
            done = is_done(next_state)
            self.update_action_value_function(state, next_state, action, reward, done)

    @abstractmethod
    def action_value_function(self, state, action):
        raise NotImplementedError

    @abstractmethod
    def action_values(self, state):
        raise NotImplementedError

    @abstractmethod
    def update_action_value_function(self, state, next_state, action, reward, done):
        raise NotImplementedError

    @abstractmethod
    def model(self, state, action):
        raise NotImplementedError

    @abstractmethod
    def update_model(self, state, action, next_state, reward):
        raise NotImplementedError

    def learn_policy(self, num_steps=10):
        """
        Learn policy with Dyna-Q algorithm.
        :param num_steps: number of steps to perform search (NB; != number of episodes)
        """

        state = self.environment.reset()
        state = tuple(converge_state(state, self.edges, self.averages))
        # keep track of performance
        running_episode_length, running_reward = 0, 0

        # loop over steps (not episodes, see Dyna-Q algorithm in Sutton p. 164)
        for _ in _tqdm(range(num_steps)):

            # perform Q-learning state on current state with current policy
            action, next_state, reward, done = self.q_learning(state)

            # save visited state-action pair
            self.add_state_action_pair(state, action)

            # model learning (deterministic or with NN)
            self.update_model(state, action, next_state, reward)

            # only do planning if enough experience gathered
            if self.pair_count >= self._planning_steps:
                self.planning()

            # check if episode done
            if not done:

                # if not done increment steps, update reward and update state
                running_episode_length += 1
                running_reward += discount_factor ** (running_episode_length - 1) * reward
                state = next_state
            else:
                # if done save episode length and reward and reset to 0, reset environment
                running_episode_length += 1
                running_reward += discount_factor ** (running_episode_length - 1) * reward
                self.episode_lengths.append(running_episode_length)
                self.total_rewards.append(running_reward)
                running_episode_length, running_reward = 0, 0
                state = self.environment.reset()
                state = tuple(converge_state(state, self.edges, self.averages))
        return

    def test_model_greedy(self, num_episodes):

        self.reset_data()

        # make policy greedy
        old_epsilon = self._epsilon
        self.set_epsilon(0)

        # loop over steps (not episodes, see Dyna-Q algorithm in Sutton p. 164)
        for _ in _tqdm(range(num_episodes)):

            done = False
            state = self.environment.reset()
            state = tuple(converge_state(state, self.edges, self.averages))

            # keep track of performance
            running_episode_length, running_reward = 0, 0

            while not done:

                # choose A from S using policy derived from Q (epsilon-greedy)
                action = self.policy_fn(state)

                # take action, observe R, S'
                (next_state, reward, done, probability) = self.environment.step(int(action))

                next_state = tuple(converge_state(next_state, self.edges, self.averages))

                running_episode_length += 1
                running_reward += discount_factor ** (running_episode_length - 1) * reward
                state = next_state

            # if done save episode length and reward and reset to 0, reset environment
            self.episode_lengths.append(running_episode_length)
            self.total_rewards.append(running_reward)

        # make policy epsilon-greedy
        self.set_epsilon(old_epsilon)

        return


class TabularDynaQ(DynaQ):

    def __init__(self, env, planning_steps=1, discount_factor=1., lr=0.5, epsilon=0.1, deterministic=True):
        super(TabularDynaQ, self).__init__(env, planning_steps, discount_factor, lr, epsilon)

        # initialize the action-value function as a nested dictionary that maps state -> (action -> action-value)
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

        # initialize the model as a nested dictionary that maps state -> (action -> (next_state, reward))
        self.det_model = defaultdict(lambda: [{"total": 0} for _ in range(env.action_space.n)])

        self.deterministic = deterministic

    def action_values(self, state):
        return self.Q[state]

    def action_value_function(self, state, action):
        return self.Q[state][action]

    def update_action_value_function(self, state, next_state, action, reward, done):

        # TODO: done is extra argument; do we need to use it in this class?

        # get max action-value
        max_action_value = max([self.Q[next_state][a] for a in range(self.environment.action_space.n)])

        # update Q
        self.Q[state][action] = self.Q[state][action] + self._learning_rate * (
                reward + discount_factor * max_action_value - self.Q[state][action]
        )

    def model(self, state, action):

        if self.deterministic:
            return self.det_model[state][action]
        else:

            choices, probabilities = [], []
            for key, values in self.det_model[state][action].items():
                if key != "total":
                    probabilities.append(values["count"] / self.det_model[state][action]["total"])
                    choices.append(key)
            next_tup_idx = np.random.choice([i for i in range(len(choices))], 1, p=probabilities)[0]
            return choices[next_tup_idx]

    def update_model(self, state, action, next_state, reward):

        if self.deterministic:
            self.det_model[state][action] = (next_state, reward)
        else:
            if (next_state, reward) not in self.det_model[state][action]:
                self.det_model[state][action][(next_state, reward)] = {}
                self.det_model[state][action][(next_state, reward)]["count"] = 1
            else:
                self.det_model[state][action][(next_state, reward)]["count"] += 1
            self.det_model[state][action]["total"] += 1


# Q: gaat het goed met tensors
class DeepDynaQ(DynaQ):

    def __init__(self, env, planning_steps=1, discount_factor=1., lr=0.5, epsilon=0.1):
        super(DeepDynaQ, self).__init__(env, planning_steps, discount_factor, lr, epsilon)

        # TODO: models finetunen
        num_hidden = 128

        # initialize neural network for Q-values
        self.Q = QNetwork(num_hidden)

        # initialize the model as a nested dictionary that maps state -> (action -> (next_state, reward))
        self.reward_model = defaultdict(lambda: [None for _ in range(env.action_space.n)])

        # initialize neural network for model of environment
        self.nn_model = ModelNetwork(num_hidden)

        # TODO: moeten we dit doen in de nn case?
        self.edges, self.averages = compute_bins([-2.4, 2.4], [-1.5, 1.5], [-0.21, 0.21], [-1.5, 1.5])

        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr)
        self.model_optimizer = optim.Adam(self.nn_model.parameters(), lr)
        self.discount_factor = discount_factor

    def action_values(self, state):
        # neural network forward function, returns action value

        # convert to PyTorch and define types
        state = torch.tensor(list(state), dtype=torch.float)

        # compute action-values of current state
        q_values = self.Q(state).detach().numpy()

        return q_values

    def action_value_function(self, state, action):
        # returns a value for a state-action pair from the NN (using self.Q)

        # compute action-values
        q_values = self.Q(state)

        # find action-value of current state-action pair
        action_value = q_values[action]

        return action_value

    def compute_target(self, reward, next_state, done):
        # done is a boolean (vector) that indicates if next_state is terminal (episode is done)

        # calculate Q values next state for batch
        action_values = self.Q(next_state)  # [B, 2]

        # get max action values for batch
        max_action_values, _ = action_values.max(dim=0)

        # calculate targets
        targets = reward + self.discount_factor * max_action_values

        # targets for terminal state equals 0
        targets[(done == 1)] = 0

        return targets

    def update_action_value_function(self, state, next_state, action, reward, done):

        # convert to PyTorch and define types
        state = torch.tensor(list(state), dtype=torch.float)
        action = torch.tensor([action], dtype=torch.int64)  # Need 64 bit to use them as index
        next_state = torch.tensor(list(next_state), dtype=torch.float)
        reward = torch.tensor([reward], dtype=torch.float)
        done = torch.tensor([done], dtype=torch.uint8)  # Boolean

        # compute the q value
        q_val = self.action_value_function(state, action)

        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = self.compute_target(reward, next_state, done)

        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(q_val, target)

        # backpropagation of loss to Neural Network (PyTorch magic)
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()

    def model(self, state, action):
        # neural network forward function, returns reward and next state

        # convert to PyTorch and define types
        state = torch.tensor(list(state), dtype=torch.float)
        action_onehot = torch.zeros(1, 2, dtype=torch.float)
        action_onehot[0][action] = 1

        # concatenate the state and action (dim=0 since we are not working with batches)
        state_action = torch.cat((state.unsqueeze(0), action_onehot), 1)

        # compute reward and next state with model network
        next_state = self.nn_model(state_action)
        state = tuple(converge_state(state, self.edges, self.averages))
        return next_state, self.reward_model[state][int(action)]

    def update_model(self, state, action, next_state, reward):
        # learn model network, gradient descent step, used in learn_policy function of base class

        # convert to PyTorch and define types
        next_state = torch.tensor(list(next_state), dtype=torch.float)
        if isinstance(state, torch.FloatTensor):
            state = state.tolist()
            state = tuple(state)
        temp_state = tuple(converge_state(state, self.edges, self.averages))
        self.reward_model[tuple(temp_state)][int(action)] = reward

        # find predicted next state and reward
        pred_next_state, _ = self.model(state, action)

        # TODO: willen we deze smooth loss?
        # compute loss
        loss_next_state = F.smooth_l1_loss(pred_next_state, next_state)
        # loss_reward = F.smooth_l1_loss(pred_reward, reward)
        loss = loss_next_state

        # backpropagation of loss to Neural Network (PyTorch magic)
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()


if __name__ == "__main__":

    # test environment
    # env = WindyGridworldEnv()
    # import gym

    env = gym.envs.make("CartPole-v0")

    # uncomment to demonstrate Q learning
    # Q_q_learning, (episode_lengths_q_learning, episode_returns_q_learning) = q_learning(env, 1000)
    #
    # # We will help you with plotting this time
    # plt.plot(episode_lengths_q_learning)
    # plt.title('Episode lengths Q-learning')
    # plt.show()
    # plt.plot(episode_returns_q_learning)
    # plt.title('Episode returns Q-learning')
    # plt.show()

    # Dyna Q
    n = 3
    learning_rate = 0.5
    discount_factor = 1
    epsilon = 0.2

    if len(sys.argv)>1 and sys.argv[1] == 'deep':
        dynaQ = DeepDynaQ(env,
                          planning_steps=n, discount_factor=discount_factor, lr=0.01, epsilon=epsilon)
    else:
        dynaQ = TabularDynaQ(env,
                             planning_steps=n, discount_factor=discount_factor, lr=learning_rate, epsilon=epsilon,
                             deterministic=False)

    dynaQ.learn_policy(10000)

    # plot results
    plt.plot(dynaQ.episode_lengths)
    plt.title('Episode lengths Deep Dyna-Q (nongreedy)')  # NB: lengths == returns
    plt.show()

    dynaQ.test_model_greedy(100)

    # plot results
    plt.plot(dynaQ.episode_lengths)
    plt.title('Episode lengths Deep Dyna-Q (greedy)')  # NB: lengths == returns
    plt.show()
