"""
Solve the multi-arm bandit problem with
- the epsilon_greedy algorithm 
"""

import numpy.random as rndm
import numpy as np

# 3 bandits with win probability of each bandit
# (win probability of each bandit is independent to other bandits, and past pulls)
p_true = [0.25, 0.50, 0.75]

# Number of trials
N_trials = 1000

# epsilon (probability of selecting a bandit at random)
epsilon = 0.10

class bandit:
    def __init__(self, p: float):
        # actual win probability of bandit
        self.p = p
        # estimated probability
        self.p_estm = 0.
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
    return rndm.choice(np.flatnonzero(arr == arr.max()))

def experiment() -> (int, float):
    bandits = [bandit(p) for p in p_true]
    for i in range(N_trials):
        if epsilon > rndm.random():
            j = rndm.randint(len(bandits))
        else:
            j = argmaxrand(np.array([b.p_estm for b in bandits]))
        # print(j, len(bandits))
        bandits[j].update()

    j_final = argmaxrand(np.array([b.p_estm for b in bandits]))

    return j_final, bandits[j_final].p_estm

bandit, bandit_p_estm = experiment()
print(bandit, bandit_p_estm)