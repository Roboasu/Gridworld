# GridWorld Q-learning with Neural Network

Neural-network-based Q-learning for the stochastic 3x4 GridWorld with experience replay and a target network. Includes a baseline run and a small hyperparameter sweep that generates plots for the report.

## Files
- assignment2.py — single-run DQN-style training + evaluation + on-screen plot
- hyper2.py — hyperparameter tuning across a small grid + smoothed training curves saved to PNG
- Report/report.tex — IEEEtran LaTeX report (figure expects smoothed_hparam_training_curves.png)

## Requirements
- Python 3.9+ (macOS)
- Packages: numpy, torch, matplotlib

Install:
- Create venv (recommended)
  - python3 -m venv .venv
  - source .venv/bin/activate
- Install deps
  - pip install numpy matplotlib torch

## Usage

Baseline training (shows a plot and prints evaluation):
- python3 assignment2.py

Hyperparameter tuning (saves plot used in the report):
- python3 hyper2.py
- Output: smoothed_hparam_training_curves.png in Assignment2/

Evaluation:
- assignment2.py prints “Evaluation Reward” and the greedy path
- hyper2.py prints a summary table with final average training reward and eval score per config

## Configuration

assignment2.py (single run):
- Episodes, gamma, epsilon schedule, lr, batch_size, target_update are defined near the top
- Actions: [N, S, E, W]
- Environment:
  - Start: (0,0)
  - Terminals: (2,3)=+1, (1,3)=-1
  - Wall: (1,1)
  - Step cost: -0.04
- To match a different start (e.g., (2,0)), change GridWorld.start

hypertuning.py (tuning):
- Sweeps: gamma in {0.9, 0.99}, epsilon in {1.0, 0.8}, lr in {1e-3, 5e-3}
- Each config trains for 1000 episodes
- Training capped at 200 steps/episode to avoid infinite loops
- Detects NaNs and aborts the config
