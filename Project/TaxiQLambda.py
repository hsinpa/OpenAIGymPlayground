import gym
import numpy as np
import random
import time

env = gym.make("Taxi-v3")

def decay_schedule(init_value, min_value, decay_ration, max_steps, log_start=-2, log_base=10):
    decay_steps = int(max_steps * decay_ration)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values-values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values * min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values


def q_lambda(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_rtatio=0.9,
             lambda_=0.5, raplacing_traces=True, n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []

    Q = np.zeros((nS, nA), dtype=np.float)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float)
    E = np.zeros((nS, nA), dtype=np.float)


q_lambda(env)