from gym import Env, spaces
from gym.utils import seeding
from dynamics import MTG, Combat, BoardState, Creature, State
import gym
import numpy as np


def invmask(l:list, mask:list):
    '''mask if False'''
    return [a for a,b in zip(l, mask) if b]

class MTGEnv(Env):
    """ The AI is player 1.
    The goal is determining whether to attack or not.
    - difficulty: max number of creatures for each player
    - nS: number of states
    - nA: number of actions
    - T: transitions (*)
    (*) dictionary of lists, where
      T[s][a] == [(probability, nextstate, reward, done), ...]
    """

    def __init__(self, difficulty=10, advantage=0):
        self.seed()
        self.difficulty = difficulty
        self.bs = MTG.gen_random_boardstate(
                c1_max=difficulty+advantage, c2_max=difficulty-advantage)
        num_c1 = len(self.bs.state1.creatures)
        num_c2 = len(self.bs.state2.creatures)
        self.nA = 2**num_c1
        self.nS = 2**(num_c1 + num_c2)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        return self.bs

    def reset(self):
        self.__init__(self.difficulty)
        return self._get_obs()

    def step(self, a):
        '''Action is a choice of attackers
        Must be an integer. Convert to base 2
        Transitions:
        T[s][a] == [(probability, nextstate, reward, done), ...]
        '''
        assert self.action_space.contains(a)
        action = MTGEnv.int_to_action(a)
        attackers = invmask(self.bs.state1.creatures, action)
        print('attackers:', attackers)
        bs_res = Combat.best_block(attackers, self.bs)
        reward = bs_res.value # - self.bs.value
        print('reward:', reward)
        self.bs = bs_res
        done = any((
                      self.bs.state1.life <= 0
                    , self.bs.state2.life <= 0
                    , len(self.bs.state1.creatures) == 0
                    , len(self.bs.state2.creatures) == 0
                    , self.bs == bs_res
                ))
        return self.bs, reward, done, {}

    @staticmethod
    def action_to_int(action:list)->int:
        ''' Converts a boolean list action to an integer.
        Ex. [True, False] -> 10 -> 2 '''
        return int(''.join([str((0,1)[i]) for i in action]), base=2)

    @staticmethod
    def int_to_action(i:int)->list:
        ''' Converts an integer into a boolean list.
        Ex. 2 -> 10 -> [True, False] '''
        return [bool(int(c)) for c in bin(i)[2:]]

    def get_random_action(self)->int:
        '''Invert is_attacking b/c masklist masks if True.
        is_attacking - boolean list of attacking status of the creatures. It's also
                        the action. '''
        is_attacking = self.np_random.binomial(len(self.bs.state1.creatures)*[1], .5)
        is_attacking = list(~np.array([bool(i) for i in is_attacking]))
        return MTGEnv.action_to_int(is_attacking)

    def get_best_action(self)->int:
        best_attack = Combat.best_attack(self.bs)
        return MTGEnv.action_to_int(best_attack)
