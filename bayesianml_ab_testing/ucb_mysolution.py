"""
Solve the multi-arm bandit problem with
- the "upper confidence bound" algorithm
    - the upper confidence bound is P(sample_mean -  true_mean >= t) <= f(t) 
    - f(t) is  a heuristic, ie. 1/t, 1/t**2, e^(-t)
    - here we use f(t) = 
    - recommended to start all bandits as if they were selected once.
"""

import numpy.random as rndm
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
        # estimated probability
        self.p_estm =  0.5
        # number of times selected
        self.n = 1

    def pull(self) -> bool:
        """ pull bandit, check if won/lost with probability p_estm """
        return self.p > rndm.random()

    def update(self):
        # update being selected
        self.n += 1
        # update estimated probability
        self.p_estm = (self.p_estm * (self.n - 1) + self.pull()) / self.n

def ucb(n:int, nj:int, mean: np.float) -> np.float:
    """ get upper confidence bound based """
    return mean + np.sqrt(2 * np.log(n) / nj)

def argmaxrand(arr: np.array) -> int:
    """argmax with random selection when max is not unique"""
    return rndm.choice(np.flatnonzero(arr == arr.max()))

def experiment() -> (int, float):
    bandits = [bandit(p) for p in p_true]
    for i in range(1, N_trials):
        j = argmaxrand(np.array([ucb(i, b.n, b.p_estm) for b in bandits]))
        bandits[j].update()

    return j, bandits[j].p_estm

bandit, bandit_p_estm = experiment()
print(bandit, bandit_p_estm)