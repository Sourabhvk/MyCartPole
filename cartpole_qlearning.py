import gymnasium as gym
import numpy as np

env = gym.make('CartPole-v1')

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)