from DQN import DQN
from Experience import Experience
from Utils import AverageRewardTracker, Logger, plot, make_env, backup_model
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np

env = make_env("Breakout-v4")
n_actions = env.action_space.n
state_dim = env.observation_space.shape
env.reset()

gamma = 0.99

epsilon = 1.0
min_epsilon = 0.1
decay_rate = 0.995 # per episode
buffer_maxlen = 100000

max_steps = 10000
batch_size = 32
train_freq = 4
target_update_freq = 10000
backup_freq = 500
reward_tracker = AverageRewardTracker(100)
logger = Logger()

agent = DQN(state_dim, n_actions, epsilon, min_epsilon, decay_rate, gamma, buffer_maxlen)

episodes = 100001
total_step_counter = 0
for episode in range(1, episodes+1):
    state = env.reset()
    print(state.shape)
    done = False
    score = 0 
    
    for step in range(max_steps):
        total_step_counter += 1
        label = f"experience {total_step_counter}"
        env.render()
        action = agent.select_action(state) 
        #print(action)
        next_state, reward, done, info = env.step(action)
        #print(next_state.shape)
        score += reward

        if step == max_steps:
            print(f"Episode reached the maximum number of steps. {max_steps}")
            done = True

        experience = Experience(state, action, reward, next_state, done, label) # create new experience object
        agent.buffer.add(experience) # add experience to buffer
        state = next_state

        if total_step_counter % target_update_freq == 0:
            print(f"Updating target model step: {step}")
            agent.update_target_weights()

        if (agent.buffer.length() >= batch_size*2) & (step % train_freq == 0):
            batch = agent.buffer.sample(batch_size)
            agent.train(batch)

        if done:
            break
    
    reward_tracker.add(score)
    average = reward_tracker.get_average()

    print(f"EPISODE {episode} finished in {step} steps, " )
    print(f"epsilon {agent.epsilon}, reward {score}. ")
    print(f"Average reward over last 100: {average} \n")
    logger.log(episode, step, score, average)

    if episode != 0 and episode % backup_freq == 0: # back up model every z steps 
      backup_model(agent.model, episode)
    
    #if total_step_counter >=50000:
    agent.epsilon_decay()

    plot(logger)
env.close()