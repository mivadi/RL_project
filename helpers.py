import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):

        # get random number
        random_number = random.uniform(0, 1)

        # get actions with maximum value
        greedy_actions = np.argwhere(Q[observation] == np.amax(Q[observation])).squeeze()
        if not len(greedy_actions.shape):
            greedy_actions = [greedy_actions]
        action = random.choice(greedy_actions)

        # if number less than epsilon, get random other actions
        if random_number <= epsilon:
            all_actions = list(range(0, nA))
            if not len(greedy_actions) == nA:
                action = random.choice(all_actions)

        return int(action)

    return policy_fn


def tabular_one_step_q(env, policy, state, Q, alpha=0.5, discount_factor=1.0):
    """
    Tabular one-step Q-learning algorithm. Takes an action according to a current state and updates the action-value
    function accordingly.


    :param env: OpenAI environment.
    :param policy: policy function that takes state and returns action
    :param state: OpenAI state
    :param Q: defaultdict of action values with keys S and A
    :param alpha: TD learning rate.
    :param discount_factor: Gamma discount factor.
    :return: Tuple of next state, observerd reward and done bool
    """

    # choose A from S using policy derived from Q (epsilon-greedy)
    action = policy(state)

    # take action, observe R, S'
    (next_state, reward, done, probability) = env.step(int(action))

    # get max action-value
    max_action_value = max([Q[next_state][a] for a in range(env.action_space.n)])

    # update Q
    Q[state][action] = Q[state][action] + alpha * (
            reward + discount_factor * max_action_value - Q[state][action]
    )

    # copy state_tilde and action_tilde to be next steps.
    state = next_state

    return action, state, reward, done


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, Q=None):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    :param env: OpenAI environment.
    :param num_episodes: Number of episodes to run for.
    :param discount_factor: Gamma discount factor.
    :param alpha: TD learning rate.
    :param epsilon: Probability to sample a random action. Float between 0 and 1.
    :param Q: hot-start the algorithm with a Q value function (optional)

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is a list of tuples giving the episode lengths and rewards.
    """

    # initialize the action-value function as a nested dictionary that maps state -> (action -> action-value)
    if Q is None:
        Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # keep track of useful statistics
    stats = []

    # the policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # loop over episodes
    for _ in _tqdm(range(num_episodes)):

        time_steps = 0
        total_return = 0

        done = False

        # initialize S
        state = env.reset()

        # loop for each step of episode
        while not done:

            _, state, reward, done = tabular_one_step_q(env, policy, state, Q, alpha, discount_factor)

            # add one time step and the rewards to the stats
            time_steps += 1
            total_return += discount_factor ** (time_steps - 1) * reward

        stats.append((time_steps, total_return))

    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

