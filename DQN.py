from utils import RLAgent
from utils import Snapshot
from experienceReplay import PrioritizedExperienceReplay
from experienceReplay import ExperienceReplay
import numpy as np
import random
import logging
from PIL import Image

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, batch_norm
from lasagne.regularization import regularize_network_params, l2
from lasagne.objectives import squared_error

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 256
# Weight decay
DECAY = 1.0e-6
# Mini-batch size
BATCH_SIZE = 32
# Discount factor gamma
GAMMA = 0.99
# Epsilon-greedy policy parameters
EPS0 = 1.0
EPS = 0.1
# Synchronise target model afterwhile
SYNC_TARGET_MODEL = 5000
# Wait till replay accumulate some experience than start learning
REPLAY_CAPACITY = 250000
MIN_REPLAY_SIZE = 50000
# Log file path
LOG_FILE = "output.log"
# Take snapshots
SNAPSHOT_EVERY = 2000
SNAPSHOT_PREFIX = 'snapshots/qmlp'
# How much prioritization to use
ALPHA = 0.6
# Bias annealing term
BETA = 0.6
# Image input size
DIMS = (1, 84, 84)
# Double Deep Q-Learning
DDQN = False

class DeepQAgent(RLAgent):
    """Q-Learning agent with Deep network function approximation"""
    def __init__(self, actions, isTerminalState, inputShape, max_iters, prediction_mode=False):
        self.actions = actions
        self.actionsNum = len(actions())
        self.inputShape = inputShape
        self.isTerminalState = isTerminalState
        self.explorationProb = EPS0
        self.numIters = 0
        self.max_iters = max_iters
        self.total_reward = 0
        self.gamma = GAMMA
        self.prediction_mode = prediction_mode
        self.start_learning = False
        self.replay = ExperienceReplay(capacity=REPLAY_CAPACITY)
        self.stats = {}
        self.frame_buf = np.zeros(DIMS)
        self.frame_num = 0

        logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
        self.snapshot = Snapshot(SNAPSHOT_PREFIX)

        self.model, self.train, self.predict = self._create_network()
        self.targetModel, self.predictTarget = self._create_fixed_network()
        self._syncModel()
        self._load_snapshot()

    def _build_network(self):
        l_in = InputLayer(shape=(None, DIMS[0], DIMS[1], DIMS[2]), name="input")
        l_1 = Conv2DLayer(l_in, num_filters=16, filter_size=(8, 8), stride=4, nonlinearity=lasagne.nonlinearities.rectify, name="conv1")
        l_2 = Conv2DLayer(l_1, num_filters=32, filter_size=(4, 4), stride=2, nonlinearity=lasagne.nonlinearities.rectify, name="conv2")
        l_3 = batch_norm(DenseLayer(l_2, num_units=N_HIDDEN, nonlinearity=lasagne.nonlinearities.rectify, name="fc1"))
        l_out = DenseLayer(l_3, num_units=self.actionsNum, nonlinearity=lasagne.nonlinearities.identity, name="out")

        return l_out, l_in.input_var

    def _create_fixed_network(self):
        print("Building network with fixed weights...")
        net, input_var = self._build_network()

        # Theano functions for training and computing cost
        print("Compiling functions ...")
        predict = theano.function([input_var], lasagne.layers.get_output(net))

        return net, predict

    def _create_network(self):
        print("Building network ...")
        net, input_var = self._build_network()
        target_values = T.matrix('target_output')
        maxQ_idx = target_values.argmax(1)

        # Create masks
        mask = theano.shared(np.ones((BATCH_SIZE, self.actionsNum)).astype(np.int32))
        maxQ_mask = theano.shared(np.zeros((BATCH_SIZE, self.actionsNum)).astype(np.int32))
        mask = T.set_subtensor(mask[np.arange(BATCH_SIZE), maxQ_idx], 0)
        maxQ_mask = T.set_subtensor(maxQ_mask[np.arange(BATCH_SIZE), maxQ_idx], 1)

        # lasagne.layers.get_output produces a variable for the output of the net
        network_output = lasagne.layers.get_output(net)
        new_target_values = target_values * maxQ_mask + network_output * mask

        err = squared_error(network_output, new_target_values)

        # Add regularization penalty
        cost = err.mean() + regularize_network_params(net, l2) * DECAY

        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(net, trainable=True)

        # Compute SGD updates for training
        updates = lasagne.updates.adadelta(cost, all_params)

        # Theano functions for training and computing cost
        print("Compiling functions ...")
        train = theano.function([input_var, target_values], [cost, new_target_values, network_output, err.mean(1), maxQ_idx], updates=updates)
        predict = theano.function([input_var], lasagne.layers.get_output(net))

        return net, train, predict

    def _syncModel(self):
        net_params = lasagne.layers.get_all_param_values(self.model)
        fixed_net_param = lasagne.layers.get_all_param_values(self.targetModel)
        diff = np.mean([np.mean(layer - fixed_net_param[i]) for i, layer in enumerate(net_params)])
        logging.debug("Syncing models, average weight diff %s" % diff)
        lasagne.layers.set_all_param_values(self.targetModel, net_params)

    def _getOptAction(self, state):
        features = self.featureExtractor(state)
        pred = self.predict(features.reshape(np.append(-1, features.shape)).astype(theano.config.floatX))
        return self.actions()[pred.argmax()]

    # Epsilon-greedy policy
    def getAction(self, state):
        self.numIters += 1
        if not self.prediction_mode and (not self.start_learning or random.random() < self.explorationProb):
            return random.choice(self.actions())
        else:
            return self._getOptAction(state)

    def provideFeedback(self, state, action, reward, new_state):
        state_features = self.featureExtractor(state)
        new_state_features = self.featureExtractor(new_state)
        self.replay.append((state_features, action, reward, new_state_features)) # Put new data to experience replay

        self.total_reward += reward
        if self.replay.getSize() >= MIN_REPLAY_SIZE:
            self.start_learning = True

        if not self.prediction_mode:
            if self.start_learning and self.replay.getSize() >= MIN_REPLAY_SIZE:
                # totalPriority = self.replay.sum
                batch = self.replay.mini_batch(BATCH_SIZE)
                target = np.zeros((BATCH_SIZE, self.actionsNum), dtype=np.float64)
                features = np.zeros(np.append(BATCH_SIZE, DIMS), dtype=np.float64)

                for i, (b_state, b_action, b_reward, b_new_state) in enumerate(batch):
                    features[i] = b_state
                    shape = np.append(-1, b_new_state.shape)

                    if self.isTerminalState(b_new_state):
                        target[i].fill(b_reward)
                    elif DDQN:
                        # Double Q-learning target
                        act = np.argmax(self.predict(b_new_state.reshape(shape).astype(theano.config.floatX)), 1)
                        q_vals = self.predictTarget(b_new_state.reshape(shape).astype(theano.config.floatX))
                        t = b_reward + self.gamma * np.ravel(q_vals)
                        target[i] = np.tile(t[act] - 1, self.actionsNum)
                        target[i][act] = t[act]
                    else:
                        q_vals = self.predictTarget(b_new_state.reshape(shape).astype(theano.config.floatX))
                        target[i] = b_reward + self.gamma * q_vals

                assert(target.shape == (BATCH_SIZE, self.actionsNum))
                loss, target_val, net_out, err, maxQ_idx = self.train(features.astype(theano.config.floatX), target.astype(theano.config.floatX))
                self.explorationProb -= (EPS0 - EPS) / (self.max_iters - MIN_REPLAY_SIZE)
                assert(np.sum(np.invert(np.isclose(target_val, net_out))) <= BATCH_SIZE)

                logging.info("Iteration: %s Replay size: %s TD-err: %s Reward: %s Action %s Epsilon: %s" %\
                                (self.numIters, self.replay.getSize(), loss, reward, action, self.explorationProb))
                logging.debug("maxQ_idx %s" % maxQ_idx)
                # keys, values = zip(*sorted(self.replay.range_stats.items(), key=lambda x: x[0][0]))
                # range_stats = np.array(values) / self.replay.range_count
                # logging.debug("Priority stats %s" % zip(keys, np.diff(range_stats)))

            if self.numIters % SYNC_TARGET_MODEL == 0:
                self._syncModel()

            if self.numIters % SNAPSHOT_EVERY == 0:
                self._save_snapshot()
        else:
            logging.info("Iteration: %s Reward: %s Action %s" % (self.numIters, reward, action))

    # def _sampling_weight(num, priority, totalPriority):
    #     weigth = (1.0 / self.maxW) * 1.0 / (num * priority / totalPriority) ** BETA
    #     if

    def _save_snapshot(self):
        snap = {"iter": self.numIters,
                "epsilon": self.explorationProb,
                "target_params": lasagne.layers.get_all_param_values(self.targetModel),
                "params": lasagne.layers.get_all_param_values(self.model),
                "replay": self.replay.experience}
        self.snapshot.save(snap, self.numIters)

    def _load_snapshot(self):
        snap = self.snapshot.load()
        if snap is not None:
            lasagne.layers.set_all_param_values(self.model, snap['params'])
            lasagne.layers.set_all_param_values(self.targetModel, snap['target_params'])
            self.numIters = snap["iter"]
            self.explorationProb = snap["epsilon"]
            self.replay.experience = snap["replay"]

    def featureExtractor(self, state):
        img = Image.fromarray(state)
        return np.array(img.resize((DIMS[1], DIMS[2]), Image.BICUBIC)).reshape(DIMS)
