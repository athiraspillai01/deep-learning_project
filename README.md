# Tetris AI with Deep Q-Learning

This project implements an AI agent that learns to play Tetris using Deep Q-Learning (DQN). The agent uses deep reinforcement learning techniques to improve its gameplay through experience.

## Project Structure

- `tetris.py`: Core Tetris game implementation with game mechanics and state management
- `dqn_agent.py`: Deep Q-Learning agent implementation with neural network architecture
- `run.py`: Main training script with hyperparameter configuration and training loop
- `test.py`: Testing utilities to evaluate trained models


## Features

### DQN Agent
- Deep Q-Network with configurable architecture
- Experience replay memory for improved learning
- Epsilon-greedy exploration strategy
- Optional target network for stable learning
- Batch normalization support

### Tetris Environment
- Custom Tetris implementation with OpenCV visualization
- State representation and reward calculation
- Comprehensive game mechanics (rotation, movement, line clearing)
- Manual and AI-controlled gameplay modes

### Training System
- Configurable hyperparameters
- TensorBoard integration for monitoring
- Automatic model saving for best performances
- Validation during training
- Performance metrics tracking

## Implementation Details

### Neural Network Architecture
```python
Model Structure:
- Input Layer: State size (game board representation)
- Hidden Layers: [256, 256] neurons with ReLU activation
- Output Layer: 1 neuron with linear activation
- Optional: Batch normalization after hidden layers
```

### Training Parameters
- Episodes: 2000
- Memory Size: 50000
- Epsilon: 1.0 → 0.1 (linear decay)
- Discount Factor: 0.98
- Batch Size: 32
- Training Frequency: Every episode
- Validation Frequency: Every 10 episodes

### Reward System
- Base reward for each move
- Additional rewards for:
  - Line clears (quadratic scaling)
  - Efficient piece placement
  - Board height management
  - Avoiding holes and bumpiness

## Usage

### Training

python run.py

### TESTING

python test.py

## Dependencies
- Python 3.6+
- TensorFlow 2.x
- NumPy
- OpenCV (cv2)
- Matplotlib
- tqdm

## Future Improvements
1. Algorithm Enhancements:
   - Prioritized experience replay
   - Double DQN implementation
   - Dueling network architecture

2. Feature Additions:
   - Advanced reward shaping
   - State representation improvements
   - Policy gradient methods
   - A3C or PPO implementations

## Results
The agent demonstrates the ability to:
- Learn effective piece placement strategies
- Clear multiple lines simultaneously
- Manage board height
- Adapt to different game situations

