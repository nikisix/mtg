from gym import Env, spaces
from gym.utils import seeding
from dynamics import *
import gym
import numpy as np


MtgEnv(Env):
    """ Has the following members
    - difficulty: max number of creatures
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    """

    def __init__(self, difficulty=10):
        self.seed()
        cs1 = self.np_random(0, difficulty) # creatures for player 1
        cs2 = self.np_random(0, difficulty) # creatures for player 2
        initial_state = gen_random_board_state(c1_max=cs1, c2_max=cs2)
        self.lastaction = None  # for rendering
        self.nA = cs1 choose 2
        self.nS = cs1+cs2 choose 2

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.lastaction = None
        # TODO

    def step(self, a):
        # TODO
        p, s, r, d = transitions[i]
        self.lastaction = a
        return (int(s), r, d, {"prob": p})
