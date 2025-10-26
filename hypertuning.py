import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque
import itertools
import time

# =====================
# 1. Environment setup
# =====================
class GridWorld:
    def __init__(self):
        self.rows, self.cols = 3, 4
        self.start = (0, 0)
        self.terminals = {(2, 3): 1.0, (1, 3): -1.0}
        self.walls = {(1, 1)}  # Wall at (1,1)
        self.rewards = {(r, c): -0.04 for r in range(self.rows) for c in range(self.cols)}
        for w in self.walls:
            if w in self.rewards:
                del self.rewards[w]
        self.rewards.update(self.terminals)
        self.actions = ['N', 'S', 'E', 'W']
        self.transition_prob = {'forward': 0.8, 'left': 0.1, 'right': 0.1}

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        if self.pos in self.terminals:
            return self.pos, self.rewards[self.pos], True

        intended_action = action
        move = self._stochastic_move(intended_action)
        next_state = self._move(self.pos, move)
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
            return dirs[(idx - 1) % 4]
        else:
            return dirs[(idx + 1) % 4]

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

# =========================
# 3. Helper functions
# =========================
def state_to_tensor(state):
    return torch.tensor(state, dtype=torch.float32)

def moving_average(data, window=20):
    """Smooths a curve using a simple moving average."""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')

def evaluate(env, q_net, episodes=5):
    total_rewards = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        total = 0
        steps = 0
        while not done and steps < 200:
            q_vals = q_net(state_to_tensor(s))
            a = env.actions[torch.argmax(q_vals).item()]
            s, r, done = env.step(a)
            total += r
            steps += 1
        total_rewards.append(total)
    return np.mean(total_rewards)

# =========================
# 4. Q-learning training
# =========================
def train_agent(gamma, epsilon, epsilon_min, epsilon_decay, lr, episodes=1000, hidden=64):
    env = GridWorld()
    q_net = QNetwork(hidden=hidden)
    target_net = QNetwork(hidden=hidden)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=5000)
    actions = env.actions

    batch_size = 64
    target_update = 50
    returns = []

    for ep in range(episodes):
        s = env.reset()
        done = False
        total_reward = 0
        steps = 0  # Prevent infinite loops

        while not done and steps < 200:
            if np.random.rand() < epsilon:
                a = random.choice(actions)
            else:
                q_vals = q_net(state_to_tensor(s))
                a = actions[torch.argmax(q_vals).item()]

            s_next, r, done = env.step(a)
            memory.append((s, actions.index(a), r, s_next, done))
            s = s_next
            total_reward += r
            steps += 1

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
                if torch.isnan(loss):
                    print(f"NaN detected at episode {ep}. Skipping this config.")
                    return returns, -999

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        if ep % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())
        returns.append(total_reward)

    return np.array(returns), evaluate(env, q_net)

# =========================
# 5. Hyperparameter tuning
# =========================
def tune_hyperparameters():
    gammas = [0.9, 0.99]
    epsilons = [1.0]
    epsilon_mins = [0.05]
    epsilon_decays = [0.99, 0.995]
    lrs = [0.0005, 0.001, 0.005]

    combinations = list(itertools.product(gammas, epsilons, epsilon_mins, epsilon_decays, lrs))
    results = []
    print(f"Total combinations: {len(combinations)}")

    start = time.time()
    for i, (gamma, epsilon, epsilon_min, epsilon_decay, lr) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Training with γ={gamma}, ε={epsilon}, ε_min={epsilon_min}, decay={epsilon_decay}, lr={lr}")
        rewards, eval_score = train_agent(
            gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay, lr=lr, episodes=1000
        )
        results.append({
            'gamma': gamma, 'epsilon': epsilon, 'epsilon_min': epsilon_min,
            'epsilon_decay': epsilon_decay, 'lr': lr,
            'train_curve': rewards, 'eval_score': eval_score,
            'final_avg_reward': np.mean(rewards[-50:]) if len(rewards) > 50 else np.mean(rewards)
        })

    print(f"\nTuning completed in {(time.time()-start)/60:.2f} minutes")
    return results

# =========================
# 6. Plot results (with smoothing)
# =========================
def plot_results(results, smooth_window=50):
    plt.figure(figsize=(8,5))
    for r in results:
        smoothed = moving_average(r['train_curve'], window=smooth_window)
        label = f"γ={r['gamma']},decay={r['epsilon_decay']},lr={r['lr']}"
        plt.plot(smoothed, label=label, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Training Reward")
    plt.title(f"Hyperparameter Tuning (Moving Avg Window={smooth_window})")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("smoothed_hparam_training_curves.png")
    plt.close('all')
    print("Plot saved as smoothed_hparam_training_curves.png")

# =========================
# 7. Main Execution
# =========================
if __name__ == "__main__":
    results = tune_hyperparameters()
    plot_results(results, smooth_window=50)

    print("\n=== Summary ===")
    for r in results:
        print(f"γ={r['gamma']}, ε={r['epsilon']}, decay={r['epsilon_decay']}, lr={r['lr']} "
              f"-> TrainAvg={r['final_avg_reward']:.3f}, Eval={r['eval_score']:.3f}")
