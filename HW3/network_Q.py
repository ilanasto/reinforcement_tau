import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load environment
env = gym.make('FrozenLake-v1', is_slippery=True)

# Define the neural network mapping a 16-dim one-hot state to 4 Q-values.
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.net(x)


model = QNetwork()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 0.1
num_episodes = 20000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s, _ = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # 1. Choose an action greedily from the Q-network
        #    (run the network for current state and choose the action with the maxQ)
        state = np.identity(16)[s:s + 1].astype(np.float32)
        state_tensor = torch.from_numpy(state)
        q_values = model(state_tensor).detach().numpy()[0]
        action = int(np.argmax(q_values))

        # 2. A chance of e to perform random action
        if np.random.rand() < e:
            action = env.action_space.sample()

        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, terminated, truncated, _ = env.step(action)
        d = terminated or truncated

        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        next_state = np.identity(16)[s1:s1 + 1].astype(np.float32)
        next_state_tensor = torch.from_numpy(next_state)
        next_q_values = model(next_state_tensor).detach().numpy()[0]

        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        target_q = q_values.copy()
        for i_q in range(len(target_q)):
            if i_q != action:
                target_q[i_q] = q_values[i_q]
            else:
                target_q[i_q] = r + y * np.max(next_q_values) * (0 if d else 1)

        # 6. Train the network using target and predicted Q values
        optimizer.zero_grad()
        predicted_q = model(state_tensor)
        loss = loss_fn(predicted_q, torch.tensor(target_q, dtype=torch.float32).unsqueeze(0))
        loss.backward()
        optimizer.step()

        rAll += r
        s = s1
        if d == True:
            #Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))
