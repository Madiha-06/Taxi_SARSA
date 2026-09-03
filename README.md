# Taxi_SARSA

A tabular **SARSA** (on-policy TD control) agent trained to solve the `Taxi-v4` environment from [Gymnasium](https://gymnasium.farama.org/), including training, evaluation with a custom OpenCV visualization, and a final greedy-policy testing phase.

## Overview

The classic Taxi problem: a taxi navigates a 5x5 grid to pick up a passenger at one of four locations (R, G, Y, B) and drop them off at another. This project implements a SARSA agent from scratch (no RL libraries) and tracks its learning progress across training, evaluation, and testing.

The three stages run end-to-end from `sarsa_taxi.py`:

1. **Training** — learns a Q-table over 5000 episodes using the SARSA update rule.
2. **Evaluation** — replays 5 episodes with the trained (greedy) policy, rendered live with a custom OpenCV top-down visualization, and plots reward/episode-length curves.
3. **Testing** — runs 10 greedy-policy episodes with no rendering and reports average reward, average episode length, and success rate.

## Project Structure

```
Taxi_SARSA/
├── sarsa_taxi.py              # Entry point — runs training, evaluation, and testing
├── sarsa_training.py          # SARSA training loop and Q-table learning
├── sarsa_evaluation.py        # Greedy-policy evaluation + OpenCV visualization
├── sarsa_testing.py           # Greedy-policy testing and success-rate reporting
├── epsilon_greedy_policy.py   # Shared epsilon-greedy action selection
├── sarsa_q_table.pkl          # Saved Q-table from a completed training run
├── requirements.txt           # Python dependencies
└── Results/                   # Training/evaluation/testing plots and a demo video
```

## Algorithm

The agent uses the standard SARSA (State-Action-Reward-State-Action) update:

```
Q(s, a) ← Q(s, a) + α [ r + γ · Q(s', a') − Q(s, a) ]
```

where `a'` is the action actually chosen by the current epsilon-greedy policy in the next state — making SARSA an on-policy method (unlike Q-learning/SARSAMAX, which bootstraps off the max action).
## Setup

- git clone https://github.com/Madiha-06/Taxi_SARSA.git
- cd Taxi_SARSA
- pip install -r requirements.txt


## Usage

Run the full training → evaluation → testing pipeline:

This will:
- Train a fresh Q-table and save it to 'sarsa_q_table.pkl'
- Display training reward/episode-length plots
- Open an OpenCV window visualizing 5 evaluation episodes using the trained policy
- Print testing metrics (average reward, average episode length, success rate) over 10 episodes and display the corresponding plots

To reuse an existing Q-table instead of retraining, comment out the training step in 'sarsa_taxi.py' and point 'q_table' to 'sarsa_q_table.pkl'.

## Results

Training, evaluation, and testing plots, along with a recorded demo of the evaluation visualization (`training ui.mp4`), are available in the ['Results/'](./Results) directory.

## Requirements

Key dependencies (see `requirements.txt` for the full pinned list):
- `gymnasium`
- `numpy`
- `opencv-python`
- `matplotlib`
