from ReplayBuffer import ReplayBuffer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import keras.backend as K
class DQN():

    def __init__(self, state_space, action_space, epsilon, min_epsilon, decay_rate, 
    gamma, buffer_maxlen):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_maxlen)
        self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        self.loss_function = keras.losses.Huber()

    def build_model(self):
    
        # Network defined by the Deepmind paper
        inputs = layers.Input(shape=(84, 84, 4))

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, (8, 8), strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, (4, 4), strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, (3, 3), strides=1, activation="relu")(layer2)
        layer4 = layers.Flatten()(layer3)
        layer5 = layers.Dense(512, activation="relu")(layer4)
        action = layers.Dense(self.action_space, activation="linear")(layer5)
       
        model = keras.Model(inputs=inputs, outputs=action)
        #model.summary()
        return model 

    def update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = np.reshape([state], (1, 84, 84, 4))
        q_vals = self.model.predict(state)
        #print(f"q values are {q_vals}")
        return np.argmax(q_vals[0])

    def train(self, experiences):
        states = []
        next_states = []
        rewards = []
        actions = []
        done = []

        for experience in experiences:
            states.append(experience.state)
            next_states.append(experience.next_state)
            rewards.append(experience.reward)
            actions.append(experience.action)
            done.append(experience.done)
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        actions = np.array(actions)
        done = np.array(done)

        future_rewards = self.target_model.predict(next_states)
        updated_q_values = rewards + self.gamma * tf.reduce_max(future_rewards, axis=1)
        
        masks = tf.one_hot(actions, self.action_space)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks),axis=1)
            loss = self.loss_function(updated_q_values, q_action)
            #print(loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # batch_q_vals_states = self.model.predict(states)
        # batch_q_vals_next_states_local = self.model.predict(next_states)

        # for index, experience in enumerate(experiences):
        #     inputs.append(experience.state)
        #     q_vals_local = batch_q_vals_next_states_local[index]
        #     print(q_vals_local)
        #     q_vals_target = batch_q_vals_next_states_target[index]
        #     print(q_vals_target)
        #     best_action_q_val = q_vals_target[np.argmax(q_vals_local)]
            
        #     if experience.done:
        #         target_val = experience.reward
        #     else:
        #         target_val = experience.reward + self.gamma + best_action_q_val
        #     #print(target_val)

        #     target_vector = batch_q_vals_states
        #     target_vector[experience.action] = target_val
        #     targets.append(target_vector)
        #     #print(target_vector)
        # return inputs, targets

    def epsilon_decay(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_rate