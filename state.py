""" run tests with:
python -m doctest ./state.py
"""
from functools import total_ordering
from hashlib import md5
from math import log
import copy
import itertools as it
import pandas as pd
import random

INF = 1000
HASHLEN = 20
DEBUG = False

def powerset(l):
    '''Produce all subsets of an iterable'''
    for sl in it.product(*[[[], [i]] for i in l]):
        yield {j for i in sl for j in i}

def masklist(mylist, mymask):
    '''mask if True'''
    return [a for a,b in zip(mylist,mymask) if not b]

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
    def __sub__(self, other): return Creature(self.p-other.p, self.t-other.t)
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

        if l1 == 0: return -1*INF
        if l2 == 0: return INF

        return (p1+t1+log(l1+1))/(p2+t2+log(l2+1))

    def __lt__(self, other):
        return self.eval() < other.eval()
    def __eq__(self, other):
        return self.eval() == other.eval()


class Combat:
    @staticmethod
    def _gen_blocks(attacker_slots: list[list[Creature]], blockers: list[Creature], possible_blocks:
            list) -> None:
        ''' Recursively generate all possible blocks.
        Note: This method generates duplicates.
        Ex. 2 attackers, 3 blockers:
        >>> Combat._gen_blocks([[],[]], list(range(2)), [])
        [[[], []], [[0], []], [[0, 1], []], [[0], [1]], [[], [0]], [[1], [0]], [[], [0, 1]], [[1], []], [[1, 0], []], [[1], [0]], [[], [1]], [[0], [1]], [[], [1, 0]]]
        '''
        # print(attacker_slots)
        possible_blocks.append(copy.deepcopy(attacker_slots))
        if len(blockers)==0: return
        for i, blocker in enumerate(blockers):
            blockers_cp = blockers.copy()
            b = blockers_cp.pop(i)
            for slot in attacker_slots:
                slot.append(b)
                Combat._gen_blocks(attacker_slots, blockers_cp, possible_blocks)
                slot.pop()
        return possible_blocks


    @staticmethod
    def _dedup_blocks(blocks: list[list[list]]):
        '''blocks: block[attacker_slot[blocker]]
        Ref block documentation for more details on the blocks data structure.
        Convert blocks to immutable and hashable types then use set to dedup

        # >>> print('\n'.join(sorted([str(e).replace('-','\t') for e in BoardState._dedup_blocks(BoardState._gen_blocks([list() for _ in range(2)], list(range(2)), list())) ])))
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
        blocks = Combat._gen_blocks(attacker_slots, blockers, blocks)
        blocks = Combat._dedup_blocks(blocks)
        return blocks

    @staticmethod
    def print_blocks(blocks):
        for block in blocks:
            for slot in block: print(f'{slot}\t', end='')
            print()

    @staticmethod
    def assign_damage(c1: Creature, cs2: set[Creature])->(Creature, list[Creature], bool):
        """ Consider permutations of damage assignment orderings
        Return: combined value of consumed blockers, optimal damage assignment order,
        and if the attacker dies
        Remember: Attacking player assigns damage.
        Score is total of consumed blocking creatures

        Can vectorize with something like:
        min(masklist(list(range(len(np.array(range(5))<3))), np.array(range(5))<3)) - 1

        >>> Combat.assign_damage(Creature(1,1), {Creature(2,2)})
        ((0,0), [], True)

        >>> Combat.assign_damage(Creature(10,10),{Creature(*t) for t in [(1,3), (5,1), (9,3), (9,5)]})
        ((23,9), [(9,5), (9,3), (5,1)], True)

        >>> Combat.assign_damage(Creature(10,10), {Creature(11,11)})
        ((0,0), [], True)

        >>> Combat.assign_damage(Creature(10,10), {Creature(4,4)})
        ((4,4), [(4,4)], False)

        """
        if DEBUG: print(c1, cs2); print('----')
        blockers_max = Creature(0,0)
        order_max = None
        for blockers in powerset(cs2):
            if len(blockers) == 0:      continue
            # Calc trade value for a blocker ordering
            ao_max = Creature(0,0)
            for assignment_order in it.permutations(blockers):
                consumed_stats = Creature(0,0)
                consumed = list()
                p = c1.p
                for b in assignment_order:
                    p -= b.t
                    if p < 0: break
                    consumed_stats += b
                    consumed.append(b)
                    if ao_max < consumed_stats:
                        ao_max = consumed_stats
                        ao_order_max = consumed
            if blockers_max < ao_max:
                blockers_max = ao_max
                order_max = ao_order_max
        #handle empty blocker set
        if order_max: order_max = sorted(order_max, reverse=True)
        else: order_max = list()
        if len(cs2)==0: attacker_dies = False
        elif blockers_max == Creature(0,0): attacker_dies = c1.t < sum(cs2).p
        else: attacker_dies = c1.t <= blockers_max.p
        return blockers_max, order_max, attacker_dies

    @staticmethod
    def attack(bs: BoardState)->set[Creature]:
        ''' Return a set of attackers if attacking is a good idea.
            1. Attack if we have lethal (<Settle The Wreckage> OP vs this AI)
            2. Consider the board-state's value after all possible blocks
        '''
        if bs.check_lethal(): return True



    @staticmethod
    def best_block(bs: BoardState)->BoardState:
        """ Evaluate all possible blocks and damage assignments thereof.
        Return resultant BoardState after the best block has been performed.

        >>> Combat.best_block(BoardState( \
            State(16, [Creature(1,1), Creature(4,2)]), \
            State(11, [Creature(2,2), Creature(2,3), Creature(8,9)])) \
        )
        16  |
        11  |       (2,2); (2,3); (8,9)

        >>> Combat.best_block( BoardState( \
            State(9, [Creature(1,5), Creature(7,9)]), \
            State(12,[Creature(*t) for t in [(2,8), (8,3), (4,8)]]) \
        ))
        9   |       (7,9)
        12  |       (2,8); (8,3); (4,8)
        """
        attackers = bs.state1.creatures
        blockers  = bs.state2.creatures

        block_vals = dict()  # TEMP

        # Use fight and state to check advantage deltas
        possible_blocks = Combat.enumerate_blocks(attackers, blockers)
        bs_min = BoardState(State(INF, []), State(-INF, []))
        if DEBUG: print(bs); print('---')
        for block in possible_blocks:
            # consumed_blocker - "cb"
            cb_stats = Creature(0,0)
            cbs = set()
            attackers_die = list()
            player2_life = bs.state2.life
            # total the blocker stats for each attacker slot in the block
            for ix_atk, s_blockers in enumerate(block.attacker_slots):
                cb_stats_i, cbs_i, attacker_dies =\
                        Combat.assign_damage(attackers[ix_atk], s_blockers)
                cb_stats += cb_stats_i
                cbs = cbs.union(set(cbs_i))
                attackers_die += [attacker_dies]
                if len(s_blockers) == 0: # attacker gets through
                    player2_life -= attackers[ix_atk].p
            state1 = State(bs.state1.life, masklist(attackers, attackers_die))
            state2 = State(player2_life, set(blockers).difference(cbs))
            bs_curr = copy.deepcopy(BoardState(state1, state2))

            if bs_curr.value < bs_min.value: bs_min = bs_curr

            if DEBUG:
                resulting_bs = [
                    bs_curr.value,
                    str(bs_curr)\
                        .replace(' ', '')\
                        .replace('\t', ' ')\
                        .replace('\n', ' || ')
                ]
                block_vals[block] = resulting_bs

        if DEBUG:
            df = pd.DataFrame(
                    data=block_vals,
                    index=it.chain(
                        [i+1 for i in range(len(attackers_die)-2)],
                        ['val'],
                        ['resulting_bs']
                    )
                ).T
            print(df.sort_values(by='val'))

        return bs_min


class MTG:
    @staticmethod
    def gen_random_board_state(c1_max=3, c2_max=3):
        num_c1=random.randint(1,c1_max)
        num_c2=random.randint(1,c2_max)
        gen_creatures = lambda max_creatures: [
                Creature(random.randint(0, 10), random.randint(1, 10))\
                    for i in range(max_creatures)]

        return BoardState(
                    State(random.randint(1, 20), gen_creatures(num_c1)),
                    State(random.randint(1, 20), gen_creatures(num_c2))
                )

    @staticmethod
    def gen_board_state_from_string(s)->BoardState:
        ''' 9       |       (1,5); (7,9)
            12      |       (2,8); (8,3); (4,8) '''
        state1str, state2str = s.split('\n')

        def gen_state_from_string(ss)->State:
            life = int(ss.split('|')[0].strip())
            creatures = [Creature(*eval(t.strip())) for t in ss.split('|')[1].strip().split(';')]
            return State(life, creatures)

        state1 = gen_state_from_string(state1str)
        state2 = gen_state_from_string(state2str)

        return BoardState(state1, state2)


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

    Combat.best_block(MTG.gen_board_state_from_string(
    ''' 9       |       (1,5); (7,9)
        12      |       (2,8); (8,3); (4,8) '''
    ))
