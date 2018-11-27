# Solution to Unity Banana Collector Environment using DQN with Prioritization Replay

## Environment Overview
This PyTorch code trains an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- move forward (0)
- move backward (1)
- turn left (2)
- turn right (3)

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.
