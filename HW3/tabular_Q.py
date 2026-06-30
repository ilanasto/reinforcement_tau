import gymnasium as gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v1', is_slippery=True)

# Implement Q-Table learning algorithm
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 20000
epsilon = 0.3
epsilon_decay = 0.9997
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s, _ = env.reset()
    rAll = 0 # Total reward during current episode
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        # 1. Choose an action by greedily (with noise) picking from Q table
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[s, :]))
        # 2. Get new state and reward from environment
        s_tplus1, reward, terminated, truncated, info = env.step(action)
        # 3. Update Q-Table with new knowledge
        max_future_q = 0.0 if terminated or truncated else np.max(Q[s_tplus1, :])
        Q[s, action] += lr * (float(reward) + y * max_future_q - Q[s, action])
        # 4. Update total reward
        rAll += float(reward)
        # 5. Move to the next state and stop if we reached a terminal state
        s = s_tplus1
        if terminated or truncated:
            break

    epsilon = max(0.01, epsilon * epsilon_decay)
    rList.append(rAll)

# Reports
print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
