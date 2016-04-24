#!/usr/bin/env python
#
import sys
import numpy as np
import argparse
from ale_python_interface import ALEInterface
from DQN import DeepQAgent

MAX_ITERS = 1000000

def init_ale(rom, display):
    ale = ALEInterface()
    # Get & Set the desired settings
    ale.setInt(b'random_seed', 123)

    # Set USE_SDL to true to display the screen. ALE must be compilied
    # with SDL enabled for this to work. On OSX, pygame init is used to
    # proxy-call SDL_main.
    USE_SDL = display
    if USE_SDL:
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            ale.setBool('sound', False) # Sound doesn't work on OSX
        elif sys.platform.startswith('linux'):
            ale.setBool('sound', True)
        ale.setBool('display_screen', display)

    # Load the ROM file
    ale.loadROM(rom)

    return ale

def solve():
    # Command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("rom", help="ROM file")
    parser.add_argument("-d", "--display", action="store_true", help="Display screen")
    parser.add_argument("-p", "--predict", action="store_true", help="Do not train the model, predict only mode")
    args = parser.parse_args()

    # Init Arcade Learning Environment
    ale = init_ale(args.rom, args.display)

    # Get the list of legal actions
    legal_actions = ale.getLegalActionSet()
    print "Legal actions %s" % legal_actions
    h, w = ale.getScreenDims()
    print "Screen dims %s" % ((h, w), )
    dims = (h, w)

    def isTerminalState(state):
        ale.game_over()

    ai = DeepQAgent(ale.getLegalActionSet, isTerminalState, dims, MAX_ITERS, args.predict)
    episode = 0
    scores = []
    total_reward = 0
    screen_state = np.empty(dims, dtype=np.uint8)
    screen_new_state = np.empty(dims, dtype=np.uint8)
    ale.getScreenGrayscale(screen_state)

    while ai.numIters <= MAX_ITERS:
        # state = ale.encodeState(ale.cloneState())
        # a = legal_actions[randrange(len(legal_actions))]
        action = ai.getAction(screen_state)
        # Apply an action and get the resulting reward
        reward = ale.act(action)
        ale.getScreenGrayscale(screen_new_state)
        ai.provideFeedback(np.copy(screen_state), action, reward, np.copy(screen_new_state))
        total_reward += reward

        if ale.game_over():
            ale.reset_game()
            ale.getScreenGrayscale(screen_state)
            scores.append(total_reward)
            print('Episode %d ended with score: %d' % (episode, total_reward))
            total_reward = 0
            episode += 1
        else:
            screen_state = np.copy(screen_new_state)

    print "Average score" % np.mean(scores)

if __name__ == "__main__":
    solve()
