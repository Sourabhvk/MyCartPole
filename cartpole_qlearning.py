import gymnasium as gym
import numpy as np

env = gym.make('CartPole-v1')

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

num_bins_pos = 6
num_bins_vel = 6
num_bins_angle = 10
num_bins_ang_vel = 10

min_pos, max_pos = -2.4, 2.4
min_angle, max_angle = -0.2095, 0.2095
min_vel, max_vel = -0.5, 0.5
min_ang_vel, max_ang_vel = -2.0, 2.0

bins_pos = np.linspace(min_pos, max_pos, num_bins_pos + 1)
bins_vel = np.linspace(min_vel, max_vel, num_bins_vel + 1)
bins_angle = np.linspace(min_angle, max_angle, num_bins_angle + 1)
bins_ang_vel = np.linspace(min_ang_vel, max_ang_vel, num_bins_ang_vel + 1)

def get_discrete_state(state):
    # Clamp values to ensure they fall within the defined bins
    clamped_pos = np.clip(state[0], min_pos, max_pos)
    clamped_vel = np.clip(state[1], min_vel, max_vel)
    clamped_angle = np.clip(state[2], min_angle, max_angle)
    clamped_ang_vel = np.clip(state[3], min_ang_vel, max_ang_vel)

    # Use np.digitize to find which bin each value falls into
    # -1 to account for 0-indexing and potential edge cases at the max bin
    discrete_pos = np.digitize(clamped_pos, bins_pos) - 1
    discrete_vel = np.digitize(clamped_vel, bins_vel) - 1
    discrete_angle = np.digitize(clamped_angle, bins_angle) - 1
    discrete_ang_vel = np.digitize(clamped_ang_vel, bins_ang_vel) - 1

    # Ensure indices are within valid range [0, num_bins - 1]
    discrete_pos = np.clip(discrete_pos, 0, num_bins_pos - 1)
    discrete_vel = np.clip(discrete_vel, 0, num_bins_vel - 1)
    discrete_angle = np.clip(discrete_angle, 0, num_bins_angle - 1)
    discrete_ang_vel = np.clip(discrete_ang_vel, 0, num_bins_ang_vel - 1)

    return (discrete_pos, discrete_vel, discrete_angle, discrete_ang_vel)

# Hyperparameters for Q-learning
learning_rate = 0.1      # Alpha (α): How much we "learn" from new information (0.0 to 1.0)
discount_factor = 0.99   # Gamma (γ): How important future rewards are compared to immediate rewards (0.0 to 1.0)
epsilon = 1.0            # Epsilon (ε): Initial exploration rate (1.0 means 100% exploration)
min_epsilon = 0.01       # Minimum exploration rate, agent will always explore at least this much
epsilon_decay_rate = 0.0001 # How quickly epsilon decreases over episodes
n_episodes = 20000        # Number of episodes to train for (adjust based on learning speed)

# Calculate the total size of the discrete state space
total_discrete_states = (num_bins_pos, num_bins_vel, num_bins_angle, num_bins_ang_vel)
# Initialize the Q-table with zeros
Q_table = np.zeros(total_discrete_states + (env.action_space.n,))

# Main training loop
for episode in range(n_episodes):
    # Reset the environment for a new episode
    current_state_continuous = env.reset()[0]
    # Convert the continuous state to our discrete representation
    current_state_discrete = get_discrete_state(current_state_continuous)

    done = False # Flag to check if the episode has ended
    rewards_current_episode = 0 # To track total reward for this episode

    # Epsilon-greedy action selection
    if np.random.rand() < epsilon:
        action = env.action_space.sample()  # Explore: select a random action
    else:
        action = np.argmax(Q_table[current_state_discrete])  # Exploit: select the best known action

    # Inner loop: Steps within an episode
    while not done:
        
        next_state_continuous, reward, done, truncated, info = env.step(action)
        rewards_current_episode += reward # Accumulate reward

        # If the episode ended (pole fell or cart went off limits)
        if done:
            break # Exit the inner while loop

    # Decay epsilon after each episode
    #epsilon = max(min_epsilon, epsilon - epsilon_decay_rate) , 
    # Decay epsilon using a multiplicative factor
    epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay_rate))

    # After the episode ends, print progress
    if episode % 1000 == 0: # Print every 100 episodes
        print(f"Episode: {episode}, Total Reward: {rewards_current_episode}, Epsilon: {epsilon:.2f}")

# Close the environment after training
env.close()
