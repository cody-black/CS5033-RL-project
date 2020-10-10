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

NUM_EPS = 5000

ENV_TYPE = "LunarLanderRandomZone"

env = gym.make('{}-v2'.format(ENV_TYPE))

q_table = np.zeros(NUM_BINS + [env.action_space.n])
# q_table = np.load("SARSA/{}/q_table.npy".format(ENV_TYPE)) # Un-comment to load q table from file

# TODO: what should these values be?
epsilon = 0.1
alpha = 0.1
gamma = 0.6

ep_rewards = []
avg_rewards = []
steps = []
success_percent = []
successes = []
final_pos = []

learn_start = time.time()
# SARASA algorithm from RL book (6.4, page 130)
# Loop for each episode
try:
    total = 0
    success_count = 0
    for i in range(NUM_EPS):
        # Initialize S
        state_i = state_to_table_indices(env.reset())

        total_reward = 0
        step_count = 0
        success = False
        done = False

        # Choose A from S using policy derived from Q
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[tuple(state_i)])

        # Loop for each step of episode
        while not done:
            # env.render() # Un-comment to render lander graphics

            # for n, value in enumerate(state):
            #     if (value == 0 or value == NUM_BINS[n] - 1) and n < 6:
            #         print("{} | {}".format(n, state))

            # Take action A, observe R, S'
            next_state, reward, done, info = env.step(action)

            if reward == 100:
                success = True
                success_count += 1

            total_reward += reward
            next_state_i = state_to_table_indices(next_state)

            # Choose A' from S' using policy derived from Q
            if np.random.uniform() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[tuple(next_state_i)])

            # Q(S, A)
            old_q_val = q_table[tuple(state_i)][action]
            # Q(S', A')
            next_q_val = q_table[tuple(next_state_i)][next_action]

            # Q(S, A) <- Q(S, A) + alpha[R + gamma * Q(S', A') - Q(S, A)]
            q_table[tuple(state_i)][action] = old_q_val + alpha * (reward + gamma * next_q_val - old_q_val)
            # S <- S'
            state_i = next_state_i
            # A <- A'
            action = next_action

            step_count += 1

        final_pos.append((next_state[0], next_state[1]))
        successes.append(success)
        steps.append(step_count)
        ep_rewards.append(total_reward)
        total += total_reward
        if i % 100 == 0:
            if i != 0:
                print("Avg. total reward Ep. {}-{}: {:.4f} | Successful landings: {}%".format(i - 99, i, total / 100, success_count))
                avg_rewards.append(total / 100)
                success_percent.append(success_count)
                total = 0
                success_count = 0
except KeyboardInterrupt: # Press CTRL+C in the terminal to stop
    pass # TODO: read in keyboard commands?

learn_stop = time.time()
learn_time = learn_stop - learn_start
np.save("SARSA/{}/q_table".format(ENV_TYPE), q_table)
np.save("SARSA/{}/num_steps".format(ENV_TYPE), steps)
np.save("SARSA/{}/rewards".format(ENV_TYPE), ep_rewards)
np.save("SARSA/{}/successes".format(ENV_TYPE), successes)
np.save("SARSA/{}/final_pos".format(ENV_TYPE), final_pos)
print("Total episodes: {} | Total time (s): {}".format(i + 1, learn_time))
print("q_table entries != 0: {}%".format(100 * np.count_nonzero(q_table) / q_table.size))
print("Steps (min, avg, max): {}, {}, {}".format(min(steps), mean(steps), max(steps)))
print("Avg time: {}/step, {}/episode".format(learn_time / sum(steps), learn_time / (i + 1)))
print("Avg time converting to indices: {}/step, {}/episode".format(to_indices_time/sum(steps), to_indices_time/(i + 1)))
fig, axes = plt.subplots(nrows=3, ncols=1)
axes[0].plot(ep_rewards)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward')
axes[1].plot(np.arange(100, i, 100), avg_rewards)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Avg Total Reward')
axes[2].plot(np.arange(100, i, 100), success_percent)
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('% successful landings')
plt.show()