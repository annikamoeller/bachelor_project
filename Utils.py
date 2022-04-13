from gym.core import ObservationWrapper
from gym.spaces import Box
from gym.core import Wrapper
import numpy as np
import gym
import cv2
from skimage.color import rgb2gray
from skimage.transform import resize
import os
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (5,5), mode='constant') * 255)
    return processed_observe

def make_env(env):
    env = gym.make(env)
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4, dim_order='tensorflow')
    return env

def backup_model(model, episode):
    backup_file = f"checkpoints/model_{episode}.h5"
    print(f"Backing up model to {backup_file}")
    model.save(backup_file)

def plot(logger):
  data = pd.read_csv(logger.file_name, sep=';')
  plt.figure(figsize=(20,15))
  plt.plot(data['average'])
  plt.plot(data['reward'])
  plt.title('Reward per training episode', fontsize=22)
  plt.xlabel('Episode', fontsize=18)
  plt.ylabel('Reward', fontsize=18)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.legend(['Average reward', 'Reward'], loc='upper left', fontsize=18)
  plt.savefig(f'metrics/reward_plot.png')

class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self,env) 
        self.img_size = (84, 84)
        self.observation_space = Box(0.0, 1.0, (self.img_size[0], self.img_size[1], 1))

    def observation(self, img):
        """what happens to each observation"""
        # crop image (top and bottom, top from 34, bottom remove last 16)
        img = img[34:-16, :, :]
        # resize image
        img = cv2.resize(img, self.img_size)
        img = img.mean(-1,keepdims=True)
        img = img.astype('float32') / 255.     
        return img

class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4, dim_order='tensorflow'):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)
        self.dim_order = dim_order
        if dim_order == 'tensorflow':
            height, width, n_channels = env.observation_space.shape
            """Multiply channels dimension by number of frames"""
            obs_shape = [height, width, n_channels * n_frames] 
        else:
            raise ValueError('dim_order should be "tensorflow" or "pytorch", got {}'.format(dim_order))
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')
        
    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer
    
    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info
    
    def update_buffer(self, img):
        if self.dim_order == 'tensorflow':
            offset = self.env.observation_space.shape[-1]
            axis = -1
            cropped_framebuffer = self.framebuffer[:,:,:-offset]
        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis = axis)

class Logger():
  def __init__(self):
    self.file_name = f'metrics/training_progress.log'
    self.reset_progress_file()

  def log(self, episode, steps, reward, average_reward):
    f = open(self.file_name, 'a')
    f.write(f"{episode};{steps};{reward};{average_reward}\n")
    f.close()

  def reset_progress_file(self):
    if os.path.exists(self.file_name):
      os.remove(self.file_name)
    f = open(self.file_name, 'a')
    f.write("episode;steps;reward;average\n")
    f.close()

class AverageRewardTracker():
  def __init__(self, episodes_to_avg=100):
    self.episodes_to_avg = episodes_to_avg
    self.tracker = deque(maxlen=episodes_to_avg)

  def add(self, reward):
    self.tracker.append(reward)

  def get_average(self):
    return np.average(self.tracker)

