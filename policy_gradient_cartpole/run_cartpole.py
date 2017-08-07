#!/usr/bin/env python
# coding=utf-8
import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
from rl_brain import policy_gradient
import matplotlib.pyplot as plt
import matplotlib
import os
#matplotlib.use('GTKAgg')
EPISODES_NUM = 200
IS_SAMPLE = True

def main():
    env = gym.make("CartPole-v0")
    env = wrappers.Monitor(env,'/tmp/cartpole-experiment-0',force=True)
    #env.monitor.start('/tmp/cartpole-experiment-0')
    agent = policy_gradient(n_actions=env.action_space.n,
                            n_features=env.observation_space.shape[0],
                            learning_rate=0.02,
                            decay=0.85,
                            is_restore=True)
    total_reward = []
    average_episode_reward = []

    for i in range(EPISODES_NUM):
        observation = env.reset()
        time_step = 0
        episode_reward = 0

        while IS_SAMPLE:
            env.render()
            action = agent.choose_action(observation);
            observation_,reward,done,info = env.step(action)
            agent.save_transition(observation,action,reward)
            episode_reward += reward
            print("episode:{0},time_step:{1},action:{2},reward:{3},done:{4}"\
                    .format(i,time_step,action,reward,done))
            if done:
                agent.learn()
                total_reward.append(episode_reward)
                average_episode_reward.append(np.mean(total_reward))
                print("Latest average_episode_reward:{0}".format(average_episode_reward[-1]))
                #if i==100:
                #    plt.plot(average_episode_reward)
                #    plt.xlabel('episode')
                #    plt.ylabel('average_episode_reward')
                #    plt.show()

                break
            time_step = time_step+1
            observation = observation_
    #env.monitor.close()
    os.system('python ./upload_openai.py')

if __name__ == "__main__":
    main()

