# gridworld_q_nn.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

# =====================
# 1. Environment setup
# =====================
class GridWorld:
    def __init__(self):
        # Grid layout (3x4)
        # Terminal states: (2,3)=+1, (1,3)=-1
        # Wall/Obstacle: (1,1)
        self.rows, self.cols = 3, 4
        self.start = (0, 0)
        self.terminals = {(2, 3): 1.0, (1, 3): -1.0}
        self.walls = {(1, 1)}
        self.rewards = {(r, c): -0.04 for r in range(self.rows) for c in range(self.cols)}
        # Remove wall from reward map (not accessible)
        for w in self.walls:
            if w in self.rewards:
                del self.rewards[w]
        # Update terminal rewards
        self.rewards.update(self.terminals)

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        if self.pos in self.terminals:
            return self.pos, self.rewards[self.pos], True

        intended_action = action
        move = self._stochastic_move(intended_action)
        next_state = self._move(self.pos, move)

        # Collision with wall → stay in same position
        if next_state in self.walls:
            next_state = self.pos

        reward = self.rewards.get(next_state, -0.04)
        done = next_state in self.terminals
        self.pos = next_state
        return next_state, reward, done

    def _move(self, state, action):
        r, c = state
        if action == 'N' and r > 0: r -= 1
        elif action == 'S' and r < self.rows - 1: r += 1
        elif action == 'E' and c < self.cols - 1: c += 1
        elif action == 'W' and c > 0: c -= 1
        return (r, c)

    def _stochastic_move(self, intended):
        dirs = ['N', 'E', 'S', 'W']
        idx = dirs.index(intended)
        roll = np.random.rand()
        if roll < 0.8:
            return intended
        elif roll < 0.9:
            return dirs[(idx - 1) % 4]  # left turn
        else:
            return dirs[(idx + 1) % 4]  # right turn

# =========================
# 2. Neural network Q model
# =========================
class QNetwork(nn.Module):
    def __init__(self, input_dim=2, output_dim=4, hidden=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ================================
# 3. Helper: state encoding & misc
# ================================
def state_to_tensor(state):
    return torch.tensor(state, dtype=torch.float32)

# ========================
# 4. Q-learning parameters
# ========================
episodes = 2000
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
lr = 0.001
batch_size = 64
target_update = 50

env = GridWorld()
q_net = QNetwork()
target_net = QNetwork()
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()
memory = deque(maxlen=5000)
actions = ['N', 'S', 'E', 'W']

# ======================
# 5. Training loop
# ======================
returns = []
for ep in range(episodes):
    s = env.reset()
    done = False
    total_reward = 0
    while not done:
        # ε-greedy exploration
        if np.random.rand() < epsilon:
            a = random.choice(actions)
        else:
            q_vals = q_net(state_to_tensor(s))
            a = actions[torch.argmax(q_vals).item()]

        s_next, r, done = env.step(a)
        memory.append((s, actions.index(a), r, s_next, done))
        s = s_next
        total_reward += r

        # Update NN after enough samples
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            s_b = torch.tensor([m[0] for m in batch], dtype=torch.float32)
            a_b = torch.tensor([m[1] for m in batch], dtype=torch.int64)
            r_b = torch.tensor([m[2] for m in batch], dtype=torch.float32)
            s_next_b = torch.tensor([m[3] for m in batch], dtype=torch.float32)
            done_b = torch.tensor([m[4] for m in batch], dtype=torch.float32)

            q_values = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_next = target_net(s_next_b).max(1)[0]
                q_target = r_b + gamma * q_next * (1 - done_b)

            loss = loss_fn(q_values, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    if ep % target_update == 0:
        target_net.load_state_dict(q_net.state_dict())
    returns.append(total_reward)

# ======================
# 6. Plot learning curve
# ======================
plt.plot(returns)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("GridWorld Q-learning with NN (wall at (1,1))")
plt.grid(True)
plt.show()

# ======================
# 7. Evaluation
# ======================
def evaluate(env, q_net):
    s = env.reset()
    done = False
    total_reward = 0
    path = [s]
    while not done:
        q_vals = q_net(state_to_tensor(s))
        a = actions[torch.argmax(q_vals).item()]
        s, r, done = env.step(a)
        path.append(s)
        total_reward += r
    return total_reward, path

reward, path = evaluate(env, q_net)
print("Evaluation Reward:", reward)
print("Path:", path)
