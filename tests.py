from dynamics import *

DEBUG = True

def test_attacks(bs_string, ans, debug=False):
    bs = MTG.gen_boardstate_from_string(bs_string)
    best_attack = Combat.best_attack(bs, DEBUG)
    if debug: print(best_attack)
    assert best_attack == ans


test1 = '''
    2       |       (1,4); (5,5); (10,8)
    17      |       (1,1); (6,9) '''
ans1 = [False, False, True]
# test_attacks(test1, ans1)

test2 = '''
    3       |       (10,10)
    17      |       (0,5); (4,1); (0,9); (3,6) '''
ans2 = [True]
# test_attacks(test2, ans2)

test3 = '''
    8       |       (1,1); (1,2); (4,7); (5,6)
    15      |       (0,3); (2,2); (6,8) '''
ans3 = [False]*4
test_attacks(test3, ans3, DEBUG)
