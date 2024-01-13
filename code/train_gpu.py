import tensorflow as tf
from utils.alpha_nnet import AlphaNNet
from utils.alpha_snake_zero_trainer import AlphaSnakeZeroTrainer

game_board_height = 15 # Standard Modus 11
game_board_width = 15 # Standard Modus 11
number_of_snakes = 4
self_play_games = 256 # Original 256
max_MCTS_depth = 16 # Original 8
max_MCTS_breadth = 128 # Original 128
initial_learning_rate = 0.0001
learning_rate_decay = 0.98

# Check if a GPU is available and use it if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Set GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU.")

name = input("Enter the model name (not including the generation number nor \".h5\"):\n")
start = int(input("Enter the starting generation (0 for creating a new model):\n"))
if start == 0:
    ANNet = AlphaNNet(input_shape=(game_board_height * 2 - 1, game_board_width * 2 - 1, 3))
    #ANNet = AlphaNNet(input_shape=(11, 11, 3))
    ANNet.save(name + "0")
else:
    ANNet = AlphaNNet(model_name="models/" + name + str(start) + ".h5")
    initial_learning_rate *= learning_rate_decay ** start

Trainer = AlphaSnakeZeroTrainer(self_play_games, max_MCTS_depth, max_MCTS_breadth,
                                initial_learning_rate, learning_rate_decay,
                                game_board_height, game_board_width, number_of_snakes, None)  # Pass None for TPU
Trainer.train(ANNet, name=name, iteration=start)

