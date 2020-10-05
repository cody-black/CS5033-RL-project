import gym
import numpy as np
import math
from statistics import mean
import matplotlib
import matplotlib.pyplot as plt
import time

# X Pos, Y Pos, X Vel, Y Vel, Angle, Angular Vel, Leg 1, Leg 2
MIN_VALS = [-2, -1, -2, -4, -math.pi / 2, -2, 0, 0] # TODO: determine realistic min/max values
MAX_VALS = [2, 2, 2, 2, math.pi / 2, 2, 1, 1]
NUM_BINS = [11, 11, 11, 11, 11, 11, 2, 2] # TODO: decide on how many bins for each value

to_indices_time = 0

def state_to_table_indices(s):
    start = time.time()
    indices = []
    for value, min_val, max_val, bins in zip(s, MIN_VALS, MAX_VALS, NUM_BINS):
        if value >= max_val:
            index = bins - 1
        elif value <= min_val:
            index = 0
        else:
            index = int(((value - min_val) * bins) // (max_val - min_val))
        indices.append(index)
    end = time.time()
    global to_indices_time
    to_indices_time += end - start
    return indices

NUM_EPS = 1000000000

env = gym.make('LunarLander-v2')

q_table = np.zeros(NUM_BINS + [env.action_space.n])
# q_table = np.load("q_table.npy") # Un-comment to load q table from file

# TODO: what should these values be?
epsilon = 0.1
alpha = 0.1
gamma = 0.6

ep_rewards = []
avg_rewards = []
steps = []

learn_start = time.time()
# Q-learning algorithm from RL book (6.5, page 131)
# Loop for each episode
try:
    total = 0
    for i in range(NUM_EPS):
        # Initialize S
        state_i = state_to_table_indices(env.reset())
        total_reward = 0
        step_cnt = 0
        done = False
        rewards = []

        # Loop for each step of episode
        while not done:
            # env.render() # Un-comment to render lander graphics

            # Choose A from S using policy derived from Q
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[tuple(state_i)])

            # for n, value in enumerate(state):
            #     if (value == 0 or value == NUM_BINS[n] - 1) and n < 6:
            #         print("{} | {}".format(n, state))

            # Take action A, observe R, S'
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            next_state_i = state_to_table_indices(next_state)

            # Q(S, A)
            old_q_val = q_table[tuple(state_i)][action]
            # max_a(Q(S', a))
            next_max = np.max(q_table[tuple(next_state_i)])

            # Q(S, A) <- Q(S, A) + alpha[R + gamma * max_a(Q(S', a)) - Q(S, A)]
            q_table[tuple(state_i)][action] = old_q_val + alpha * (reward + gamma * next_max - old_q_val)
            # S <- S'
            state = next_state_i

            step_cnt += 1

        steps.append(step_cnt)
        ep_rewards.append(total_reward)
        total += total_reward

        if i % 100 == 0:
            if i != 0:
                print("Average total reward Ep. {}-{}: {}".format(i - 99, i, total / 100))
                avg_rewards.append(total / 100)
            # print("Episode {} total reward: {}".format(i, total_reward))
            total = 0
except KeyboardInterrupt: # Press CTRL+C in the terminal to stop
    pass # TODO: read in keyboard commands?

learn_stop = time.time()
learn_time = learn_stop - learn_start
np.save("q_table", q_table)
print("Total episodes: {} | Total time (s): {}".format(i, learn_time))
print("Percent of q_table entries != 0: {}".format(100 * np.count_nonzero(q_table) / q_table.size))
print("Steps (min, avg, max): {}, {}, {}".format(min(steps), mean(steps), max(steps)))
print("Avg time: {}/step, {}/episode".format(learn_time / sum(steps), learn_time / i))
print("Avg time converting to indices: {}/step, {}/episode".format(to_indices_time/sum(steps), to_indices_time/i))
fig, axes = plt.subplots(nrows=2, ncols=1)
axes[0].plot(ep_rewards)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward')
axes[1].plot(avg_rewards)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Avg Total Reward')
plt.show()