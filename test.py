import os
import glob
from tetris import Tetris
from dqn_agent import DQNAgent
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.models import load_model

def find_best_model():
    """Find the model file with the highest score in the models directory."""
    model_files = glob.glob('models/tetris-*.h5')  # Changed pattern to match training model names
    if not model_files:
        raise FileNotFoundError("No model files found in the models directory!")
    
    # Get the most recently created model file
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model

def test_model(num_games=5, render=True, delay=0.1):
    """
    Test the best model for a specified number of games.
    
    Args:
        num_games (int): Number of games to play
        render (bool): Whether to display the game
        delay (float): Delay between moves (seconds) for visualization
    """
    # Initialize environment and agent
    env = Tetris()
    state_size = env.get_state_size()
    agent = DQNAgent(state_size)
    
    # Load the best model
    try:
        model_path = find_best_model()
        print(f"Loading model from: {model_path}")
        agent.model = load_model(model_path)  # Use Keras load_model instead
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Statistics tracking
    scores = []
    pieces = []
    lines_cleared = []
    
    print("\nStarting test games...")
    for game in range(num_games):
        current_state = env.reset()
        done = False
        game_score = 0
        game_pieces = 0
        game_lines = 0
        
        while not done:
            # Get next action
            next_states = env.get_next_states()
            next_actions, next_states = zip(*next_states.items())
            next_states = np.array(next_states)
            
            # Predict Q values and select best action
            predictions = []
            for state in next_states:
                predictions.append(agent.predict_value(np.array([state])))
            index = np.argmax(predictions)
            action = next_actions[index]
            
            # Perform action
            x, rotation = action
            reward, done = env.play(x, rotation, render=render, render_delay=delay)
            game_score += reward
            game_pieces += 1
            
            # Track lines cleared (reward > 1 means lines were cleared)
            if reward > 1:
                lines = int((reward - 1) ** 0.5)  # Reverse the reward calculation
                game_lines += lines
            
            if render and delay:
                time.sleep(delay)  # Add delay for visualization
        
        scores.append(game_score)
        pieces.append(game_pieces)
        lines_cleared.append(game_lines)
        print(f"Game {game + 1}/{num_games}:")
        print(f"  Score: {game_score}")
        print(f"  Pieces Placed: {game_pieces}")
        print(f"  Lines Cleared: {game_lines}")
    
    # Print summary statistics
    print("\nTest Results:")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Average Pieces per Game: {np.mean(pieces):.2f}")
    print(f"Max Pieces: {np.max(pieces)}")
    print(f"Average Lines Cleared: {np.mean(lines_cleared):.2f}")
    print(f"Max Lines Cleared: {np.max(lines_cleared)}")

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = f"test_results_{timestamp}.txt"
    with open(results_file, 'w') as f:
        f.write("Test Results:\n")
        f.write(f"Model used: {model_path}\n")
        f.write(f"Number of games: {num_games}\n\n")
        f.write("Per Game Results:\n")
        for i in range(num_games):
            f.write(f"Game {i+1}:\n")
            f.write(f"  Score: {scores[i]}\n")
            f.write(f"  Pieces Placed: {pieces[i]}\n")
            f.write(f"  Lines Cleared: {lines_cleared[i]}\n")
        f.write("\nSummary Statistics:\n")
        f.write(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}\n")
        f.write(f"Max Score: {np.max(scores)}\n")
        f.write(f"Average Pieces per Game: {np.mean(pieces):.2f}\n")
        f.write(f"Max Pieces: {np.max(pieces)}\n")
        f.write(f"Average Lines Cleared: {np.mean(lines_cleared):.2f}\n")
        f.write(f"Max Lines Cleared: {np.max(lines_cleared)}\n")
    print(f"\nDetailed results saved to: {results_file}")

def test_movement():
    """Test piece movement in all directions."""
    env = Tetris()
    current_state = env.reset()
    
    print("Testing piece movement...")
    
    # Test left movement
    initial_x = env.current_pos[0]
    env.move_left()
    assert env.current_pos[0] == initial_x - 1 or env.current_pos[0] == initial_x, "Left movement failed"
    print("Left movement: OK")
    
    # Test right movement
    initial_x = env.current_pos[0]
    env.move_right()
    assert env.current_pos[0] == initial_x + 1 or env.current_pos[0] == initial_x, "Right movement failed"
    print("Right movement: OK")
    
    # Test downward movement
    initial_y = env.current_pos[1]
    env.move_down()
    assert env.current_pos[1] == initial_y + 1 or env.current_pos[1] == initial_y, "Down movement failed"
    print("Down movement: OK")
    
    print("\nMovement tests completed successfully!")

if __name__ == "__main__":
    # Test movement mechanics
    test_movement()
    
    # Original test code
    NUM_GAMES = 5      # Number of games to play
    RENDER = True      # Whether to show the game
    DELAY = 0.1       # Delay between moves (seconds)
    
    test_model(NUM_GAMES, RENDER, DELAY)
