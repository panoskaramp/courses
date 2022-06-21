"""
Solve the multi-arm bandit problem with
- the "bayessian" approach
    - the reward is a bernouli distribution
    - the conjugate prior of it is beta distribution
    - recommended to start all bandits as if they were selected once
    - thomson sampling for selection of bandit
"""

import numpy.random as rndm
from numpy.random import normal
from numpy.random import beta
import numpy as np


# 3 bandits with win probability of each bandit
# (win probability of each bandit is independent to other bandits, and past pulls)
p_true = [0.25, 0.50, 0.75]

# Number of trials
N_trials = 10000

class bandit:
    def __init__(self, p: float):
        # actual win probability of bandit
        self.p = p
        # beta coefficients
        self.a =  1
        self.b =  1
        # number of times selected
        self.n = 1

    def pull(self) -> bool:
        """ pull bandit, check if won/lost with probability p_estm """
        return self.p > beta(self.a, self.b)

    def update(self, x):
        # update being selected
        self.n += 1
        # update beta coefficients
        self.a = self.a + x
        self.b = self.b + 1 - x

def argmaxrand(arr: np.array) -> int:
    """argmax with random selection when max is not unique"""
    return rndm.choice(np.flatnonzero(arr == arr.max()))

def experiment() -> (int, int, int):
    bandits = [bandit(p) for p in p_true]
    for i in range(1, N_trials):
        j = argmaxrand(np.array([beta(bndt.a, bndt.b) for bndt in bandits]))
        bandits[j].update(bandits[j].pull())

    return j, bandits[j].a, bandits[j].b

bndt, a, b = experiment()
print(bndt)

mn = 0
N = 10000
for i in range(N):
    mn += beta(a,b)
print(mn/N)