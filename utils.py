# imports

import socket
import cPickle as pickle
import glob
import os

# -------------------------------------------------- #

class RLAgent:
    """Abstract class: an RLAgent performs reinforcement learning.
    The game client will call getAction() to get an action, perform the action, and
    then provide feedback (via provideFeedback()) to the RL algorithm, so it can learn."""
    def getAction(self, state):
        raise NotImplementedError("Override me")

    def provideFeedback(self, state, action, reward, new_state):
        raise NotImplementedError("Override me")

# -------------------------------------------------- #

class Snapshot:
    def __init__(self, prefix):
        self.prefix = prefix

    def save(self, state, num):
        file_path = '_'.join([self.prefix, str(num)])
        f = file(''.join([file_path, '.pkl']), 'wb')
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self):
        file_path = glob.glob(''.join([self.prefix, "*_[0-9]*.pkl"]))
        file_path.sort(key=os.path.getctime)
        state = None
        if len(file_path):
            f = file(file_path[-1], 'rb')
            print "Loading snapshot", file_path[-1]
            state = pickle.load(f)
            f.close()

        return state
