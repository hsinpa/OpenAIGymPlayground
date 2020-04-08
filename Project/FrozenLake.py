import gym
import numpy as np

env = gym.make("FrozenLake-v0")

observationSpaceNum = env.observation_space.n
actionSpaceNum = env.action_space.n

print("Space num " + str(observationSpaceNum))
print("Action num " + str(actionSpaceNum))

iteration = 1000
gamma = 0.9

def value_iteration(env, gamma = 1.0):
    value_table = np.zeros(observationSpaceNum)

    for i in range(iteration):
        update_value_table = np.copy(value_table)

        for state in range(observationSpaceNum):
            Q_value = []

            for action in range(actionSpaceNum):
                next_states_rewards = []

                for next_st in env.unwrapped.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_st
                    value = (trans_prob * (reward_prob + gamma * update_value_table[next_state]))
                    next_states_rewards.append(value)
                Q_value.append(np.sum(next_states_rewards))
                value_table[state] = max(Q_value)

        threshold = 1e-20
        if (np.sum(np.fabs(update_value_table - value_table)) <= threshold) :
            print('Value - iteration converged at iteration # %d.' %(i+1))
            break

    return value_table

def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
            policy[state] = np.argmax(Q_table)
    return policy

value_table = value_iteration(env, gamma)
policy_table = extract_policy(value_table, gamma)

print(policy_table)