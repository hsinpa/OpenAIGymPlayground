import gym
import numpy as np
import random
import time

env = gym.make("Taxi-v3")

def select_action(state, Q, epsilon):
    if random.uniform(0, 1) < epsilon:
        return np.random.randint(len(Q[state]))
    else:
        #Loop though actionNum, and pass int(x) to lambda
        return np.argmax(Q[state])

def decay_schedule(init_value, min_value, decay_ration, max_steps, log_start=-2, log_base=10):
    decay_steps = int(max_steps * decay_ration)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values-values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values * min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values


def sara_lambda(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_rtatio=0.9,
             lambda_=0.5, raplacing_traces=True, n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []

    Q = np.zeros((nS, nA), dtype=np.float)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float)
    E = np.zeros((nS, nA), dtype=np.float)

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_rtatio, n_episodes)

    for e in range(n_episodes):
        E.fill(0)
        state, done = env.reset()
        action = select_action(state, Q, epsilons[e])

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])

            td_target = reward + gamma * Q[next_state][next_state] * (not done)
            td_error = td_target - Q[state][action]

            E[state][action] = E[state][action] + 1
            if raplacing_traces:
                E.clip(0, 1, out=E)

            Q = Q + alphas[e] * td_error * E
            E = gamma * lambda_ * E

            state, action = next_state, next_action

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis = 1)


sara_lambda(env=env)
