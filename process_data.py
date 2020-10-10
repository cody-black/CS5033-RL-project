import numpy as np
from statistics import mean
import matplotlib
import matplotlib.pyplot as plt

METHODS = ["SARSA", "Q-learning"]
ENV_NAME = "LunarLander"
AVG_SIZE = 250

running_avg_reward = dict.fromkeys(METHODS)
avg_successes = dict.fromkeys(METHODS)
avg_steps = dict.fromkeys(METHODS)

for method in METHODS:
    # Load values for each RL method
    rewards = np.load("{}/{}/rewards.npy".format(method, ENV_NAME))
    successes = np.load("{}/{}/successes.npy".format(method, ENV_NAME))
    steps = np.load("{}/{}/num_steps.npy".format(method, ENV_NAME))
    running_avg_reward[method] = []
    avg_successes[method] = []
    avg_steps[method] = []
    # Calculate the running average of the total reward for each episode
    for i in range(len(rewards)):
        running_avg_reward[method].append(mean(rewards[max(0, i - AVG_SIZE - 1):(i+1)]))
        avg_successes[method].append(100*np.count_nonzero(successes[max(0, i - AVG_SIZE - 1):(i+1)]) / min(AVG_SIZE, i + 1))
        avg_steps[method].append(mean(steps[max(0, i - AVG_SIZE - 1):(i+1)]))

fig, axis = plt.subplots()
for method in METHODS:
    axis.plot(running_avg_reward[method], label=method)
axis.set_xlabel("Episode")
axis.set_ylabel("Avg. Total Reward")
plt.legend(loc="lower right")

fig, axis = plt.subplots()
for method in METHODS:
    axis.plot(avg_successes[method], label=method)
axis.set_xlabel("Episode")
axis.set_ylabel("Avg. Successes (%)")
plt.legend(loc="lower right")

fig, axis = plt.subplots()
for method in METHODS:
    axis.plot(avg_steps[method], label=method)
axis.set_xlabel("Episode")
axis.set_ylabel("Avg. Steps")
plt.legend(loc="lower right")

plt.show()