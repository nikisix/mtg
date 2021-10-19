""" run tests with:
python -m doctest ./state.py
"""
from functools import total_ordering
from hashlib import md5
import copy
import itertools as it
import random

INF = 100
HASHLEN = 20

def powerset(l):
    '''Produce all subsets of an iterable'''
    for sl in it.product(*[[[], [i]] for i in l]):
        yield {j for i in sl for j in i}

@total_ordering
class Creature:
    def __init__(self, power: int, toughness: int):
        self.power = power
        self.p = power
        self.toughness = toughness
        self.t = toughness
        self._hash = int(md5(random.randbytes(HASHLEN)).hexdigest(), base=16)
        self.abilities = None  #TODO{'flying': False, 'first_strike': True}
    def __hash__(self): return self._hash
    def __repr__(self): return f'({self.p},{self.t})'
    def __eq__(self, other):  return (self.p+self.t) == (other.p+other.t)
    def __lt__(self, other):  return (self.p+self.t) <  (other.p+other.t)
    def __add__(self, other): return Creature(self.p+other.p, self.t+other.t)
    def __radd__(self, other):
        if      other == 0: return self
        else:   return self.__add__(other)


class Block:
    '''[[2, 1], [0]]'''
    def __init__(self, attacker_slots: list[list[Creature]]):
        self.attacker_slots = attacker_slots
    def __repr__(self):     return '-'.join([str(set(slot)) for slot in self.immute()])
    def immute(self):
        '''Convert the list of lists to tuples for hashing'''
        return tuple(tuple(slot) for slot in self.attacker_slots)
    def __hash__(self):     return self.__repr__().__hash__()
    def __str__(self):      return self.__repr__()
    def __eq__(self, other):return self.__hash__() == other.__hash__()


class State:
    def __init__(self, life: int, creatures: list[Creature]):
        '''Extend with: (num_cards_in_hand, cards_in_hand)'''
        self.life = life
        self.creatures = sorted(creatures)

    def __repr__(self):
        creatures_str = "; ".join([str(c) for c in self.creatures])
        return f'{self.life}\t|\t{creatures_str}'

    def power_sort(self):
        pows = [c.p for c in self.creatures]
        return sorted(pows, reverse=True)


@total_ordering
class BoardState:
    """Evaluates the BoardState from the perspective of player 1
        and assigns to self.value. """
    def __init__(self, state1, state2):
        self.state1 = state1
        self.state2 = state2
        self.value = self.eval()

    def __repr__(self):
        return '\n'.join([str(self.state1), str(self.state2)])

    def check_lethal(self):
        """Returns:
        1 - player 1 has lethal
        0 - else
        """
        pows1 = self.state1.power_sort()
        num_cs2 = len(self.state2.creatures)

        return self.state2.life <= sum(pows1[num_cs2:])

    def eval(self):
        '''
        score = (power+toughness)/opp_life_total
        value = score1/score2

        Can we win?
        Can we sacrifice and win next turn?
        Can opponent win on the crack-back?
        '''
        if self.check_lethal(): return INF

        p1 = sum([c.p for c in self.state1.creatures])
        p2 = sum([c.p for c in self.state2.creatures])

        t1 = sum([c.t for c in self.state1.creatures])
        t2 = sum([c.t for c in self.state2.creatures])

        l1 = self.state1.life
        l2 = self.state2.life

        return ((p1+t1)/l2)/((p2+t2)/l1)

    def __lt__(self, other):
        return self.eval() < other.eval()
    def __eq__(self, other):
        return self.eval() == other.eval()


class Combat:
    @staticmethod
    def block(attacker_slots: list[list[Creature]], blockers: list[Creature], possible_blocks:
            list) -> None:
        ''' Recursively generate all possible blocks.
        Note: This method generates duplicates.
        Ex. 2 attackers, 3 blockers:
        >>> BoardState.block([[],[]], list(range(2)), [])
        [[[], []],
         [[0], []],
         [[0, 1], []],
         [[0], [1]],
         [[], [0]],
         [[1], [0]],
         [[], [0, 1]],
         [[1], []],
         [[1, 0], []],
         [[1], [0]],
         [[], [1]],
         [[0], [1]],
         [[], [1, 0]]]
        '''
        # print(attacker_slots)
        possible_blocks.append(copy.deepcopy(attacker_slots))
        if len(blockers)==0: return
        for i, blocker in enumerate(blockers):
            blockers_cp = blockers.copy()
            b = blockers_cp.pop(i)
            for slot in attacker_slots:
                slot.append(b)
                Combat.block(attacker_slots, blockers_cp, possible_blocks)
                slot.pop()
        return possible_blocks


    @staticmethod
    def dedup_blocks(blocks: list[list[list]]):
        '''blocks: block[attacker_slot[blocker]]
        Ref block documentation for more details on the blocks data structure.
        Convert blocks to immutable and hashable types then use set to dedup
        >>> s.add((tuple(b[0]), tuple(b[1])))

        >>> set(tuple(blockers) for blockers in b)
        {(0, 1), (2,)}

        >>> print('\n'.join(sorted([str(e).replace('-','\t') for e in BoardState.dedup_blocks(BoardState.block([list() for _ in range(2)], list(range(2)), list())) ])))
        set()   set()
        set()   {0, 1}
        set()   {0}
        set()   {1}
        {0, 1}  set()
        {0}     set()
        {0}     {1}
        {1}     set()
        {1}     {0}
        '''
        s_blocks = set()
        for block in blocks: s_blocks.add(Block(block))
        return s_blocks


    @staticmethod
    def enumerate_blocks(attackers: list[Creature], blockers: list[Creature]):
        """ Say we have three attackers and three blockers [5, 6, 7]
        for each attacker we can subdivide the blockers like:
        [5,6,7]; [5],[6,7]; [5],[6],[7]; [5, 6],[7]; [5,7],[6]
        """
        attacker_slots = [list() for _ in range(len(attackers))]
        blocks = list()
        blocks = Combat.block(attacker_slots, blockers, blocks)
        blocks = Combat.dedup_blocks(blocks)
        return blocks

    @staticmethod
    def print_blocks(blocks):
        for block in blocks:
            for slot in block: print(f'{slot}\t', end='')
            print()


    @staticmethod
    def attack(bs: BoardState)->bool:
        """ Return True if attacking is a good idea.
        1. Attack if we have lethal (<Settle The Wreckage> OP vs this AI)
        2. Consider the board-state's value after all possible blocks
        """
        attackers = bs.state1.creatures
        blockers  = bs.state2.creatures

        if bs.check_lethal(): return True
        # Use fight and state to check advantage deltas
        possible_blocks = Combat.enumerate_blocks(attackers, blockers)
        bs_cur_val  = bs.eval()
        bs_star_val = bs.eval()
        for block in blocks:
            for ix_atk, s_blockers in enumerate(block):
                bs_next_val = assign_damage(attackers[ix_atk], s_blockers)
                if bs_star_val < bs_next_val: bs_star_val = bs_next_val


class MTG:
    @staticmethod
    def assign_damage(c1: Creature, c2: Creature)->int:
        """
        >>> assign_damage(Creature(1, 2), Creature(1, 1))
        1

        >>> assign_damage(Creature(1, 1), Creature(1, 1))
        0

        >>> assign_damage(Creature(1, 1), Creature(2, 2))
        -1
        """
        a1 = (c1.p >= c2.t)  #a - "advantage"
        a2 = (c2.p >= c1.t)

        if      a1 and a2:          return  0  # both die
        elif    a1 and (not a2):    return  1  # only c2 dies
        else:                       return -1  # only c1 dies

    @staticmethod
    def assign_damage(c1: Creature, cs2: set[Creature])->int:
        # TODO redo from perspective of attacker. Factor in total stats, not just the
        # toughness
        """ Currently a greedy algorithm based on the presence/absence of hanging damage.
        Remember: Attacking player assigns damage; scores relative to creature 1 (attacker).
        >>> MTG.assign_damage(Creature(4, 4), {Creature(1, 1), Creature(1,1)})
        1

        >>> MTG.assign_damage(Creature(2, 2), {Creature(1, 1), Creature(1,1)})
        0

        >>> MTG.assign_damage(Creature(2, 2), {Creature(1, 1), Creature(2,2)})
        0

        >>> MTG.assign_damage(Creature(4, 4), {Creature(3, 3), Creature(2,2)})
        -1

        TODO: Should be 1 bc more total stats are off the table
        >>> MTG.assign_damage(Creature(4, 4), {Creature(10, 3), Creature(2,2)})
        -1

        >>> MTG.assign_damage(Creature(3, 3), {Creature(4, 4)})
        -1
        """
        even = lambda c1, cs2: c1.p == sum(cs2).t

        def bad_trade(c1, cs2):
            # TODO consider permutations return a resource amount
            consumed = Creature(0,0)
            p = c1.p
            # import pdb; pdb.set_trace()  # TODO BREAKPOINT
            for b in sorted(blockers, reverse=True):
                p -= b.t
                if   p == 0:return consumed+b < c1 #False
                elif p < 0: return consumed   < c1 #True
                consumed += b
            return consumed < c1 #False

        res=1
        print(c1, cs2)
        print('----')
        for blockers in powerset(cs2):
            if len(blockers) == 0:      continue
            print(blockers)
            print('bad trade', bad_trade(c1, blockers))
            if bad_trade(c1, blockers): return -1
            if even(c1, blockers):      res = 0
        return res

    @staticmethod
    def gen_random_board_state(num_c1=3, num_c2=2):
        gen_creatures = lambda max_creatures: [
                Creature(random.randint(0, 10), random.randint(1, 10))\
                    for i in range(max_creatures)]

        return BoardState(
                    State(random.randint(1, 20), gen_creatures(num_c1)),
                    State(random.randint(1, 20), gen_creatures(num_c2))
                )


if __name__ == '__main__':
    c1 = Creature(1, 2)
    c2 = Creature(1, 1)
    s1 = State(20, [c1, c1])
    s2 = State(20, [c2, c2])
    bs = BoardState(s1, s2)
    bs.eval()

    # bs = MTG().gen_random_board_state()
    # l1 = list(range(5))
    # l2 = [5,6,7]
    # [5,6,7]; [5],[6,7]; [5],[6],[7]; [5, 6],[7]; [5,7],[6]

    # from state import *
    # bs = MTG.gen_random_board_state()
    # cs1 = bs.state1.creatures
    # cs2 = bs.state2.creatures
    # Combat.enumerate_blocks(cs1, cs2)
