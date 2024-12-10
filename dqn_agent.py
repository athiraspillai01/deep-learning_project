import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization
from collections import deque
import numpy as np
import random

# Deep Q Learning Agent + Maximin
#
# This version only provides only value per input,
# that indicates the score expected in that state.
# This is because the algorithm will try to find the
# best final state for the combinations of possible states,
# in constrast to the traditional way of finding the best
# action for a particular state.
class DQNAgent:

    '''Deep Q Learning Agent + Maximin

    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon (float): Exploration (probability of random values given) value at the start
        epsilon_min (float): At what epsilon value the agent stops decrementing it
        epsilon_stop_episode (int): At what episode the agent stops decreasing the exploration variable
        n_neurons (list(int)): List with the number of neurons in each inner layer
        activations (list): List with the activations used in each inner layer, as well as the output
        loss (obj): Loss function
        optimizer (obj): Otimizer used
        replay_start_size: Minimum size needed to train
    '''

    def __init__(self, state_size, mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_episode=500,
                 n_neurons=[32,32], activations=['relu', 'relu', 'linear'],
                 add_batch_norm=False,use_target_model=False,update_target_every=None,
                 loss='mse', optimizer='adam', replay_start_size=None):

        assert len(activations) == len(n_neurons) + 1

        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_stop_episode
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        self.model = self._build_model(add_batch_norm)
        
        self.use_target_model = use_target_model
        self.target_counter = 0  # Initialize target_counter for all cases
        self.update_target_every = update_target_every
        if use_target_model:
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_counter = 0

    def _build_model(self, add_batch_norm=False):
        '''Builds a Keras deep neural network model'''
        model = Sequential()
        model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))
        
        if add_batch_norm:
            model.add(BatchNormalization())
        
        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))
            if add_batch_norm:
                model.add(BatchNormalization())
                
        model.add(Dense(1, activation=self.activations[-1]))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        
        return model


    def add_to_memory(self, current_state, next_state, reward, done):
        '''Adds a play to the replay memory buffer'''
        self.memory.append((current_state, next_state, reward, done))


    def random_value(self):
        '''Random score for a certain action'''
        return random.random()


    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict(state, verbose=0)[0]


    def act(self, state):
        '''Returns the expected score of a certain state'''
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)


    def best_state(self, states, exploration=True):
        '''Returns the best state for a given collection of states'''
        if exploration and random.random() <= self.epsilon:
            return random.choice(list(states))

        else:
            states = np.array(list(states))
            values = self.model.predict(states, verbose=0)
            best_state = states[np.argmax(values)]
            
        return list(best_state)


    def train(self, memory_batch_size=32, training_batch_size=32, epochs=3):
        '''Trains the agent'''
        n = len(self.memory)
    
        if n >= self.replay_start_size and n >= memory_batch_size:

            # Get a random batch from memory
            batch = random.sample(self.memory, memory_batch_size)

            # Create the Q-value target
            states = np.array([x[0] for x in batch])
            new_states = np.array([x[1] for x in batch])
            
            # Use target model if enabled
            if self.use_target_model:
                new_predictions = self.target_model.predict(new_states, batch_size=training_batch_size, verbose=0)
            else:
                new_predictions = self.model.predict(new_states, batch_size=training_batch_size, verbose=0)
                
            old_predictions = self.model.predict(states, batch_size=training_batch_size, verbose=0)

            # Create the training batch and train the model
            x = []
            y = []
            for i in range(len(batch)):
                state_r, new_state_r, reward_r, done_r = batch[i]
                old_value = old_predictions[i]
                if done_r:
                    new_value = reward_r
                else:
                    new_value = reward_r + self.discount * new_predictions[i][0]  # Get scalar value
                x.append(state_r)
                y.append([new_value])  # Make it a single-element array
                
            x_train = np.array(x)
            y_train = np.array(y)
            
            history = self.model.fit(x_train, y_train,
                                   batch_size=training_batch_size,
                                   epochs=epochs,
                                   verbose=0)
            loss = history.history['loss'][-1]
            
            # Update target model if needed
            if self.use_target_model and self.update_target_every is not None:
                self.target_counter += 1
                if self.target_counter >= self.update_target_every:
                    self.target_model.set_weights(self.model.get_weights())
                    self.target_counter = 0
            
            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
            
            return loss
        return 0
