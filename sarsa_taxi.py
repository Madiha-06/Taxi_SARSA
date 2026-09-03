import gymnasium as gym

from sarsa_testing import Tester
from sarsa_training import Training
from sarsa_evaluation import Evaluator

# deterministic
taxi_env = gym.make('Taxi-v4',is_rainy=False)

# Training
train_sarsa = Training.training(taxi_env)

# Evaluation
q_table = "sarsa_q_table.pkl"
evaluation=Evaluator(taxi_env,q_table)
evaluation.visualize()

# Testing
tester=Tester(taxi_env,q_table)
tester.test()


