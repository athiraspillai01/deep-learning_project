#%% Initialize
from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
import cv2
from time import time
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os

def get_movement_reward(env, action):
    """Calculate additional reward based on movement quality"""
    x, rotation = action
    reward = 1.0  # Base reward for each move
    
    # Reward for using different positions (encourage movement)
    board_center = env.BOARD_WIDTH // 2
    distance_from_center = abs(x - board_center)
    
    # Higher reward for using full width of board
    if distance_from_center >= 2:
        reward += 1.0  # Significant reward for using edges
    else:
        reward += 0.5  # Moderate reward for center
    
    # Bonus for rotation (encourage piece manipulation)
    if rotation > 0:
        reward += 0.5
    
    # Evaluate piece placement
    current_piece = env._get_rotated_piece()
    test_pos = [x, env.current_pos[1]]
    if not env._check_collision(current_piece, test_pos):
        temp_board = env._add_piece_to_board(current_piece, test_pos)
        
        # Check for line clears (major reward)
        lines_cleared, _ = env._clear_lines(temp_board)
        if lines_cleared > 0:
            reward += lines_cleared * 2.0
        
        # Height management
        sum_height, max_height, min_height = env._height(temp_board, env._column_heights(temp_board))
        if max_height < env.BOARD_HEIGHT * 0.6:  # Keeping height lower
            reward += 1.0
        
        # Holes and bumpiness (small penalties)
        holes = env._number_of_holes(temp_board, env._column_heights(temp_board))
        total_bumpiness, _ = env._bumpiness(temp_board, env._column_heights(temp_board))
        
        reward -= (holes * 0.2)  # Smaller hole penalty
        reward -= (total_bumpiness * 0.1)  # Smaller bumpiness penalty
    
    return max(reward, 0)  # Ensure reward is never negative

def experience(render=False, max_steps=200):
    current_state = env.reset()
    done = False
    steps = 0
    total_movement_reward = 0
    
    while not done and (not max_steps or steps < max_steps):
        next_states = env.get_next_states()
        best_state = agent.best_state(next_states.values(), exploration=True)
        
        best_action = None
        for action, state in next_states.items():
            if state == best_state:
                best_action = action
                break

        # Calculate movement-based reward
        movement_reward = get_movement_reward(env, best_action)
        total_movement_reward += movement_reward
        
        # Play the move and get game reward
        game_reward, done = env.play(best_action[0], best_action[1], render=render,
                                   render_delay=None)
        
        # Combine rewards
        total_reward = game_reward + movement_reward
        
        agent.add_to_memory(current_state, next_states[best_action], total_reward, done)
        current_state = next_states[best_action]
        steps += 1
    
    return total_movement_reward

def validate(num_reps, validate_render=True, max_steps=200):
    val_scores_record = []
    val_steps_record = []
    val_movement_scores = []
    
    for rep in range(num_reps):
        if validate_render and not rep:
            render = True
        else:
            render = False
            
        current_state = env.reset()
        done = False
        steps = 0
        total_movement_reward = 0
        
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values(), exploration=False)
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break
            
            # Track movement quality
            movement_reward = get_movement_reward(env, best_action)
            total_movement_reward += movement_reward
    
            reward, done = env.play(best_action[0], best_action[1], render=render,
                                  render_delay=None)
            
            current_state = next_states[best_action]
            steps += 1
        
        val_scores_record.append(env.get_game_score())
        val_steps_record.append(steps)
        val_movement_scores.append(total_movement_reward)
    
    mean_score = np.mean(val_scores_record)
    mean_steps = np.mean(val_steps_record)
    mean_movement = np.mean(val_movement_scores)
    
    print()
    print("---------------")
    print("Validation")
    print()
    print("Max Score:\t" + str(max(val_scores_record)))
    print("Game Score:\t" + str(mean_score))
    print("Movement Score:\t" + str(mean_movement))
    print("Steps:\t\t" + str(mean_steps))
    print("---------------")
    print()
    
    return val_scores_record, val_steps_record


def dqn():        
    best_score = float('-inf')
    best_model_path = None
    last_100_scores = deque(maxlen=100)

    for episode in tqdm(range(1,episodes+1), desc="Training Progress"):
        # Game
        experience()
        current_score = env.get_game_score()
        scores.append(current_score)
        last_100_scores.append(current_score)
        
        # Train
        if episode % train_every == 0:
            loss = agent.train(memory_batch_size=memory_batch_size, training_batch_size=training_batch_size, epochs=epochs)
            if loss:
                print(f"\nEpisode {episode} - Loss: {loss:.4f} - Epsilon: {agent.epsilon:.4f}")
    
        # Validate
        if episode % validate_every == 0:
            print(f"\nValidation at episode {episode}")
            val_score, val_step = validate(num_val_reps, validate_render, max_steps)
            val_scores.append(np.mean(val_score))
            val_steps.append(np.mean(val_step))
            
            # Track best score and save best model
            current_max = max(val_score)
            if current_max > best_score:
                best_score = current_max
                print(f"\nNew best score: {best_score}")
                
                # Remove previous best model if it exists
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
                # Save new best model
                best_model_path = f'models/tetris_best_model_score_{best_score:.0f}.h5'
                print(f"Saving new best model to {best_model_path}")
                agent.model.save(best_model_path)
    
        wall_time.append(time())
        
        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])
            val_avg_score = mean(val_score)
            val_min_score = min(val_score)
            val_max_score = max(val_score)
            
            print(f"\nLast {log_every} episodes - Avg: {avg_score:.1f}, Min: {min_score}, Max: {max_score}")
            print(f"Last 100 episodes - Avg: {mean(last_100_scores):.1f}")
            
            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score, val_avg_score=val_avg_score,
                    val_min_score=val_min_score, val_max_score=val_max_score,
                    wall_time=time() - start_time)


#%% Models
env = Tetris()
episodes = 2000
max_steps = 200
epsilon = 1.0
epsilon_min = 0.1
epsilon_stop_episode = 1000  # Faster epsilon decay
mem_size = 50000  # Even larger memory
discount = 0.98
training_batch_size = 32  # Smaller batches for more frequent updates
memory_batch_size = 32
epochs = 4  # More epochs per training
replay_start_size = 500  # Start training earlier
train_every = 1

validate_every = 10
num_val_reps = 5
validate_render = True
log_every = 10  # More frequent logging

n_neurons = [256, 256]
activations = ['relu', 'relu', 'linear']
add_batch_norm = True
use_target_model = True
update_target_every = 50  # More frequent target updates

agent = DQNAgent(env.get_state_size(),
                 n_neurons=n_neurons, activations=activations, add_batch_norm=add_batch_norm,
                 epsilon=epsilon, epsilon_min=epsilon_min,use_target_model=use_target_model,
                 update_target_every=update_target_every,
                 epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                 discount=discount, replay_start_size=replay_start_size)

log_dir = f'logs/tetris-epsilon={epsilon}-epsilon_min={epsilon_min}-epsilon_stop_episode={epsilon_stop_episode}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
log = CustomTensorBoard(log_dir=log_dir)

start_time = time()
wall_time = []
scores = []
val_scores = []
val_steps = []

dqn()


#%% Analyze

plt.plot(range(len(val_scores)), val_scores)
plt.plot(range(len(scores)), scores)
