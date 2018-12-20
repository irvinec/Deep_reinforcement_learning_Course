"""
Download roms from: http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html

extract all the "Roms.rar" files.
python -m retro.import <Path to Roms>
"""

import tensorflow as tf
#from tensorflow.python import debug as tf_debug
import numpy as np
import random
import retro

# For image preprocessing
from skimage import transform
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

from collections import deque
from collections import namedtuple

# ignore some warnings from skimage
import warnings

# Image processing hyperparameters
FRAME_STACK_SIZE = 4

# Training hyperparameters
NUM_EPISODES = 50
MAX_NUM_STEPS = 50000
BATCH_SIZE = 64

# Exploration hyperparameters
MAX_EXPLORE_PROB = 1.0
MIN_EXPLORE_PROB = 0.1
EXPLORE_RATE_DECAY = 0.00001    # Exponential decay rate for exploration probability

# Q learning hyperparameter
DISCOUNT_RATE = 0.9
LEARNING_RATE = 0.00025     # Alpha (aka learning rate)

# Memory hyperparameters
NUM_PRETRAIN_STEPS = BATCH_SIZE
MEMORY_SIZE = 1000000


def main():
    env = retro.make('SpaceInvaders-Atari2600')
    print(f'The size of our frame is: {env.observation_space.shape}')
    print(f'The number of actions: {env.action_space.n}')

    # Create a one-hot encoded version of our actions
    one_hot_actions = np.identity(env.action_space.n, dtype=int)

    # Initialize deque with zero-images one array for each image
    stacked_frames = create_stacked_frames()

    # Setup the first pre-train episode
    state = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, state, True) # is new episode

    

    # Initialize DQN and memory replay buffer
    
    memory_buffer = MemoryBuffer(max_size=MEMORY_SIZE)

    # Pretrain/Prefill the memory buffer
    for _ in range(NUM_PRETRAIN_STEPS):
        # Take a random action
        random_action_index = random.randint(0, len(one_hot_actions) - 1)
        random_action = one_hot_actions[random_action_index]
        next_state, reward, done, _ = env.step(random_action)
        env.render()

        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False) # is not new episode

        # If the episode is finished (we are dead 3x)
        if done:
            # Zero out the next state
            next_state = np.zeros(state.shape)

            # Add the experience to memory
            memory_buffer.add(Experience(state, random_action, reward, next_state, done))

            # Reset for the start of a new episode
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True) # is new episode
        else:
            memory_buffer.add(Experience(state, random_action, reward, next_state, done))
            state = next_state

    # Reset the computation graph (just in case)
    tf.reset_default_graph()
    # TODO: do these names really need to match e.g. DQN to DQN
    dqn = DQN(state.shape, env.action_space.n)

    # Set up logging and Saving
    writer = tf.summary.FileWriter('./tensorboard/DQN/1')
    tf.summary.scalar('Loss', dqn.loss)
    write_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    # train
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        for episode_index in range(NUM_EPISODES):
            episode_reward = 0
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True) # is new episode
            for step_index in range(MAX_NUM_STEPS):
                # Pick an action and take it
                action, explore_probability = select_action(state, one_hot_actions, step_index, dqn, sess)
                next_state, reward, done, _ = env.step(action)
                env.render()
                episode_reward += reward
                if done:
                    # Add a terminal a frame 
                    next_state = np.zeros((110, 84), dtype=int)
                    next_state, stacked_frames(stacked_frames, next_state, False) # is not a new episode

                    print(f'Episode: {episode_index + 1}')
                    print(f'Total Reward for Episode: {episode_reward}')
                    print(f'Explore Probability: {explore_probability:.4f}')
                    print(f'Training Loss: {loss:.4f}')

                    # Store transition/experience in memory <s_t, a_t, r_t+1, s_t+1>
                    memory_buffer.add(Experience(state, action, reward, next_state, done))

                    # We are done with this episode, so exit the loop
                    break
                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, state, False) # is not a new episode
                    memory_buffer.add(Experience(state, action, reward, next_state, done))
                    state = next_state

                ### LEARNING PART
                # Obtain random mini-batch of Experiences from memory
                batch = memory_buffer.sample(BATCH_SIZE)
                # TODO is there a better way to slice on tuple?
                # ndmin=3 for states since they are 3D tensors (array of 2D frames)
                states_mini_batch = np.array([exp.state for exp in batch], ndmin=3)
                actions_mini_batch = np.array([exp.action for exp in batch])
                next_states_mini_batch = np.array([exp.next_state for exp in batch], ndmin=3)

                target_q_values_batch = []

                # Get Q values for next state
                q_values_next_state = sess.run(dqn.output, feed_dict={dqn.inputs: next_states_mini_batch})
                for experience_index, experience in enumerate(batch):
                    if experience.done:
                        target_q_values_batch.append(experience.reward)
                    else:
                        target = experience.reward + DISCOUNT_RATE*np.max(q_values_next_state[experience_index])
                        target_q_values_batch.append(target)

                    loss, _ = sess.run(
                        [dqn.loss, dqn.optimizer],
                        feed_dict={
                            dqn.inputs: states_mini_batch,
                            dqn.target_q: target_q_values_batch,
                            dqn.actions: actions_mini_batch
                        }
                    )

                    summary = sess.run(
                        write_op,
                        feed_dict={
                            dqn.inputs: states_mini_batch,
                            dqn.target_q: target_q_values_batch,
                            dqn.actions: actions_mini_batch
                        }
                    )
                    writer.add_summary(summary, episode_index)
                    writer.flush()

                    # Save model, every 5 episodes
                    if episode_index % 5 == 0:
                        _ = saver.save(sess, "./models/model.ckpt")
    # end training

def create_stacked_frames():
    return deque([np.zeros((110,84), dtype=int) for i in range(FRAME_STACK_SIZE)], maxlen=FRAME_STACK_SIZE)

def preprocess_frame(frame):
    # Greyscale frame
    gray = rgb2gray(frame)

    # Crop the screen (remove the part below the player)
    cropped_frame = gray[8:-12, 4:-12]

    # Normalize pixel values
    normalized_frame = cropped_frame/255.0

    # Resize to 110x84
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])

    # preprocessed frame is 110x84x1
    return preprocessed_frame

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        # TODO: make this take in an optional param to fill
        # the frames.
        stacked_frames = create_stacked_frames()
        # Replicate the first frame to fill the stacked frames
        stacked_frames.extend([frame]*FRAME_STACK_SIZE)
    else:
        # Append the frame to the deque, automatically removes the oldest frame.
        stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

def select_action(state, actions, decay_step, dqn, sess):
    explore_probability = MIN_EXPLORE_PROB + (MAX_EXPLORE_PROB - MIN_EXPLORE_PROB)*np.exp(-EXPLORE_RATE_DECAY*decay_step)
    if explore_probability > np.random.rand():
        # Pick a random action
        action = actions[random.randint(0, len(actions) - 1)]
    else:
        # Get the action from the DQN
        q_values = sess.run(dqn.output, feed_dict={dqn.inputs, state.reshape((1, *state.shape))})
        # Get the best action from the larget argmax Q(s,a)
        action_index = np.argmax(q_values)
        action = actions[action_index]
    return action, explore_probability

class DQN(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope('DQN'):
            # We create the placeholders
            # NOTE: *state_size unpacks the tuple

            self.inputs = tf.placeholder(tf.float32, [None, *self.state_size], name='inputs')
            self.actions = tf.placeholder(tf.float32, [None, self.action_size], name='actions')

            # Remember that target_Q is the R(s,a) + ymax*Qhat(s',a')
            # TODO: Review the target network section
            self.target_q = tf.placeholder(tf.float32, [None], name='target')

            # Build the network
            # First convolution layer with ELU activation
            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding='VALID',
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name='conv1'
            )
            self.conv1_out = tf.nn.elu(self.conv1, name='conv1_out')

            # Second convolutional layer with ELU activation.
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding='VALID',
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name='conv2'
            )
            self.conv2_out = tf.nn.elu(self.conv2, name='conv2_out')

            # Third convolutional layer with ELU activation
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=64,
                kernel_size=[3, 3],
                strides= [2, 2],
                padding='VALID',
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name='conv3'
            )
            self.conv3_out = tf.nn.elu(self.conv3, name='conv3_out')

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            self.fully_connected = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='fully_connnected'
            )

            self.output = tf.layers.dense(
                inputs=self.fully_connected,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=self.action_size,
                activation=None
            )

            # Our predicted Q value
            self.predicted_q = tf.reduce_sum(tf.multiply(self.output, self.actions))
            # The loss is the sum of (target_q - predicted_q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_q - self.predicted_q))
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class MemoryBuffer(object):
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Take batch_size samples from the buffer
        return random.sample(self.buffer, batch_size)

if __name__ == "__main__":
    main()