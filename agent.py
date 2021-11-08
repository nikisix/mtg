import env
from statistics import mean

VERBOSE=True

supported_agents = {'random1', 'random2', 'best'}
AGENT='best'
assert AGENT in supported_agents

env = env.MTGEnv(difficulty=4, advantage=3)
agent_funcs = {
        'random1': env.action_space.sample,
        'random2': env.get_random_action,
        'best': env.get_best_action
    }

rewards = list()
for i_episode in range(20):
    ep_reward=list()
    observation = env.reset()
    for t in range(100):
        if VERBOSE: print(observation)
        action = agent_funcs[AGENT]()
        observation, reward, done, info = env.step(action)
        if reward != 0: ep_reward.append(reward)
        if done:
            if len(ep_reward)==0: ep_reward = [0]
            ep_reward = mean(ep_reward)
            if VERBOSE:
                print(observation)
                print("Episode finished after {} timesteps".format(t+1))
                print(f"Episodic Avg Reward: {ep_reward}")
                print("------------------------------------------------")
            rewards.append(ep_reward)
            break
print(f"{AGENT} agent Avg Reward: {mean(rewards)}")
env.close()
