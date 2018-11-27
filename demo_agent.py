from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
from dqn_agent import Agent


env = UnityEnvironment(file_name="./Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)
print('States have length:', state_size)


#-----------------------------------------------------------------------------------------------------------------------
# Load weights for DQN agent

agent = Agent(state_size, action_size, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('solution.pt'))


#-----------------------------------------------------------------------------------------------------------------------
# Take actions in the environment

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]  # get the current state
score = 0  # initialize the score
while True:
    action = agent.act(state, eps=0.0)  # select an action
    env_info = env.step(action)[brain_name]  # send the action to the environment
    next_state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0]  # get the reward
    done = env_info.local_done[0]  # see if episode has finished
    score += reward  # update the score
    state = next_state  # roll over the state to next time step
    if done:  # exit loop if episode finished
        break

print("Score: {}".format(score))