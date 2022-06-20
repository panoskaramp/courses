"""
Solve the multi-arm bandit problem with
- the "optimistic" algorithm (start with a very high win probability estimate (p_estm))
    - the bandit selected will then get a lower p_estm, and perhaps wont be the max for the next round
    - need to start all bandits as if they were selected once.
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
        self.p_estm = 1000. #####
        # number of times selected
        self.n = 0

    def pull(self) -> bool:
        """ pull bandit, check if won/lost with probability p_estm """
        return self.p > rndm.random()
    
    def update(self):
        # update being selection
        self.n += 1
        # update estimated probability
        self.p_estm = (self.p_estm * (self.n - 1) + self.pull()) / self.n

def argmaxrand(arr) -> int:
    """argmax with random selection when max is not unique"""
    return rndm.choice(np.flatnonzero(arr == arr.max()))

def experiment() -> (int, float):
    bandits = [bandit(p) for p in p_true]
    for i in range(N_trials):
        j = argmaxrand(np.array([b.p_estm for b in bandits]))
        # print(j, len(bandits))
        bandits[j].update()

    j_final = argmaxrand(np.array([b.p_estm for b in bandits]))

    return j_final, bandits[j_final].p_estm

bandit, bandit_p_estm = experiment()
print(bandit, bandit_p_estm)