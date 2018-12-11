from model import TabularDynaQ, DeepDynaQ
import sys
import gym
from helpers import ReplayMemory, smooth
import matplotlib.pyplot as plt


env = gym.envs.make("CartPole-v0")

# Dyna Q
n = 10
learning_rate = 0.5
discount_factor = .8
epsilon = 0.2
capacity = 10000
experience_replay = True
true_gradient = False
batch_size = 1

if len(sys.argv) > 1 and sys.argv[1] == 'deep':
    memory = ReplayMemory(capacity)
    dynaQ = DeepDynaQ(env,
                      planning_steps=n, discount_factor=discount_factor, lr=1e-3, epsilon=epsilon, memory=memory,
                      experience_replay=experience_replay, true_gradient=true_gradient, batch_size=batch_size)
else:
    dynaQ = TabularDynaQ(env,
                         planning_steps=n, discount_factor=discount_factor, lr=learning_rate, epsilon=epsilon,
                         deterministic=False)

dynaQ.learn_policy(1000)

# plot results
plt.plot(smooth(dynaQ.episode_lengths, 10))
plt.title('Episode lengths Deep Dyna-Q (nongreedy)')  # NB: lengths == returns
plt.show()

dynaQ.test_model_greedy(100)

# plot results
plt.plot(smooth(dynaQ.episode_lengths, 10))
print("Average episode length (greedy): {}".format(np.mean(np.array(dynaQ.episode_lengths))))
plt.title('Episode lengths Deep Dyna-Q (greedy)')  # NB: lengths == returns
plt.show()
