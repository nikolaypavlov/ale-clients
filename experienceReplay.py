import numpy as np
from blist import sortedset
from collections import Counter

class ExperienceReplay:
    """Stores all trajectories during the life of the agent"""

    def __init__(self, min_usages=8, capacity=100):
        self.experience = []
        self.min_usages = min_usages
        self.capacity = capacity

    def append(self, exp):
        self.experience.append((exp, 0))

    def mini_batch(self, size):
        idx = np.random.choice(range(len(self.experience)), size, replace=False)
        batch = [None] * size
        removelist = []
        for i, n in enumerate(idx):
            exp, num = self.experience[n]
            num += 1
            batch[i] = exp

            if num >= self.min_usages:
                removelist.append(n)

            self.experience[n] = (exp, num)

        if self.getSize() >= self.capacity:
            for i in sorted(removelist, reverse=True):
                del self.experience[i]

        return batch

    def getSize(self):
        return len(self.experience)

class PrioritizedExperienceReplay():
    def __init__(self, alpha=0, min_usages=8, capacity=4096, batch_size=32):
        self.sum = 0
        self._alpha = alpha
        self.min_usages = min_usages
        self.capacity = capacity
        self.experience = sortedset(key=lambda x: x[0])
        self.maxP = np.finfo(np.float32).eps
        self.batch_size = batch_size
        assert(capacity % batch_size == 0)
        _ranges = np.floor(np.linspace(0, self.capacity, self.batch_size + 1)).astype(np.int)
        self.ranges = zip(_ranges, _ranges[1:])[::-1]
        self.range_stats = Counter()
        self.range_count = 0

    def clean_up(self):
        while self.getSize() < self.capacity:
            self.experience.pop(0)

    def append(self, state, action, reward, next_state, count=0, td_err=None):
        if self.getSize() >= self.capacity:
            self.clean_up()

        if count < self.min_usages:
            if td_err is not None:
                priority = (abs(td_err) + np.finfo(np.float32).eps) ** self._alpha
            else:
                priority = self.maxP

            self.maxP = max(self.maxP, priority)
            transition = (priority, (state, action, reward, next_state), count)
            self.experience.add(transition)
            self.sum += priority

    def mini_batch(self):
        batch = [None] * self.batch_size

        for i, (start, end) in enumerate(self.ranges):
            sample_idx = (np.random.choice(np.arange(start, end), 1, replace=False))
            priority, (state, action, reward, next_state), count = self.experience.pop(sample_idx)
            batch[i] = ((state, action, reward, next_state), count + 1, priority)
            self.range_stats[(start, end)] += priority / self.sum
            self.sum -= priority
        self.range_count += 1

        return batch

    def getSize(self):
        return len(self.experience)
