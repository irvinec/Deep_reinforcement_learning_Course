"""
Play space invaders using random actions.

Created mainly to debug/PoC taking actions in space invaders game.
"""
import numpy as np
import random
import retro

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

NUM_EPISODES = 2
MAX_NUM_STEPS = 50000

def main():
    env = retro.make('SpaceInvaders-Atari2600')
    # Create a one-hot encoded version of our actions
    one_hot_actions = np.identity(env.action_space.n, dtype=int)

    # Setup the first episode
    for episode_index in range(NUM_EPISODES):
        state = env.reset()
        env.render()
        
        done = False
        # Take steps until the episode ends or we reach max steps
        for _ in range(MAX_NUM_STEPS):
            # Take a random action
            random_action_index = random.randint(0, len(one_hot_actions) - 1)
            random_action = one_hot_actions[random_action_index]
            next_state, reward, done, _ = env.step(random_action)
            env.render()
            if done:
                print(f'Episode {episode_index} reached end state.')
                break

        if not done:
            print(f'Episode {episode_index} reached max steps.')

if __name__ == "__main__": main()