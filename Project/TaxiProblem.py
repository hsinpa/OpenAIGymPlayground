import gym
import numpy as np
import random
import time

env = gym.make("Taxi-v3")

observationNum = env.observation_space.n
actionNum = env.action_space.n

print("Action num ", actionNum)

alpha = 0.3
gamma = 0.98
epsilon = 0.08
episode = 10000
q = {}
#initialization
for s in range(observationNum):
    for a in range(actionNum):
        q[(s,a)] = 0.0

def update_q_table(prev_state, action, reward, nextstate,alpha, gamma):
    qa = max([q[(nextstate, a)] for a in range(actionNum) ])
    q[(prev_state, action)] += alpha * (reward + gamma * qa - q[(prev_state, action)])

def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        #Loop though actionNum, and pass int(x) to lambda
        return max(list(range(actionNum)), key = lambda x: q[(state,x)])

def train(episode):
    for i in range(episode):
        r = 0

        prev_state = env.reset()

        while True:
            action = epsilon_greedy_policy(prev_state, epsilon)
            nextstate, reward, done, _ = env.step(action)

            update_q_table(prev_state, action, reward, nextstate, alpha, gamma)

            prev_state = nextstate
            r += reward

            if done:
                break

        print("Reward", r)


train(episode)
env.close()
