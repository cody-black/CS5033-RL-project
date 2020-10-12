# CS 5033 Short RL Project
##### Cody Black and Ray Niazi

---

### Q-learning.py and SARSA.py
Running one of these programs will train an agent using the chosen RL method. 

The type of Lunar Lander environment can be changed by changing ENV_NAME.

Changing NUM_EPS will change the number of episodes the agent is trained for. The training will automatically end after the specified number of episodes or can be stopped early by pressing CTRL+C in the terminal.

When the agent finishes the specified number of episodes or is stopped early with CTRL+C, the agent's data will be saved.

### evaluate_agents.py

Running this program runs each agent on each environment for NUM_EPS episodes. The type of agents are specified by the METHODS list and the environments are specified by the ENVS list. However, the program will automatically skip any agent/environment combination that it does not have data for.

The program calculates the average reward, success percentage, and steps per episode over NUM_EPS episodes and prints them to the terminal. It also saves these values to a .csv file.

### plot_data.py

This program creates plots for average reward, success percentage, and steps per episode. The averages are calculated at each episode using the values for the previous AVG_SIZE episodes. The type of agents are specified by the METHODS list and the environments are specified by the ENVS list. However, the program will automatically skip any agent/environment combination that it does not have data for.

The figures generated are saved in the "Figures" directory, which should be created by the program if it does not already exist.

### lunar_lander.py

This is a modified version of the default [OpenAI Gym Lunar Lander environment](https://gym.openai.com/envs/LunarLander-v2/). The modifications include the addition of different variants of the Lunar Lander environment and a modification to the reward function that reduces the reward proportional to the magnitude of the lander's angular velocity.

Lunar Lander variants:

* **LunarLander:** Default Lunar Lander environment. The lander starts with a random velocity above a stationary landing zone in the middle of the screen.
* **LunarLanderMovingZone:** The landing zone is moving from left to right over flat terrain.
* **LunarLanderMoreRandStart:** Same as LunarLander, but the maximum random starting velocity is increased.
* **LunarLanderRandomZone:** The landing zone is stationary but in a random location.
* **LunarLanderLimitedFuel:** Same as LunarLander, but the lander has a limited amount of fuel. When the fuel is exhausted, attempting to fire the engines no longer has any effect.