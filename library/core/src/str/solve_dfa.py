#!/usr/bin/env python3
# Use z3 to solve UTF-8 validation DFA for offset and transition table,
# in order to encode transition table into u32.
# We minimize the output variables in the solution to make it deterministic.
# Ref: <https://gist.github.com/dougallj/166e326de6ad4cf2c94be97a204c025f>
# See more detail explanation in `./validations.rs`.
#
# It is expected to find a solution in <30s on a modern machine, and the
# solution is appended to the end of this file.
from z3 import *

STATE_CNT = 9

# The transition table.
# A value X on column Y means state Y should transition to state X on some
# input bytes. We assign state 0 as ERROR and state 1 as ACCEPT (initial).
# Eg. first line: for input byte 00..=7F, transition S1 -> S1, others -> S0.
TRANSITIONS = [
    # 0  1  2  3  4  5  6  7  8
    # First bytes
    ((0, 1, 0, 0, 0, 0, 0, 0, 0), "00-7F"),
    ((0, 2, 0, 0, 0, 0, 0, 0, 0), "C2-DF"),
    ((0, 3, 0, 0, 0, 0, 0, 0, 0), "E0"),
    ((0, 4, 0, 0, 0, 0, 0, 0, 0), "E1-EC, EE-EF"),
    ((0, 5, 0, 0, 0, 0, 0, 0, 0), "ED"),
    ((0, 6, 0, 0, 0, 0, 0, 0, 0), "F0"),
    ((0, 7, 0, 0, 0, 0, 0, 0, 0), "F1-F3"),
    ((0, 8, 0, 0, 0, 0, 0, 0, 0), "F4"),
    # Continuation bytes
    ((0, 0, 1, 0, 2, 2, 0, 4, 4), "80-8F"),
    ((0, 0, 1, 0, 2, 2, 4, 4, 0), "90-9F"),
    ((0, 0, 1, 2, 2, 0, 4, 4, 0), "A0-BF"),
    # Illegal
    ((0, 0, 0, 0, 0, 0, 0, 0, 0), "C0-C1, F5-FF"),
]

o = Optimize()
offsets = [BitVec(f"o{i}", 32) for i in range(STATE_CNT)]
trans_table = [BitVec(f"t{i}", 32) for i in range(len(TRANSITIONS))]

# Add some guiding constraints to make solving faster.
o.add(offsets[0] == 0)
o.add(trans_table[-1] == 0)

for i in range(len(offsets)):
    # Do not over-shift. It's not necessary but makes solving faster.
    o.add(offsets[i] < 32 - 5)
    for j in range(i):
        o.add(offsets[i] != offsets[j])
for trans, (targets, _) in zip(trans_table, TRANSITIONS):
    for src, tgt in enumerate(targets):
        o.add((LShR(trans, offsets[src]) & 31) == offsets[tgt])

# Minimize ordered outputs to get a unique solution.
goal = Concat(*offsets, *trans_table)
o.minimize(goal)
print(o.check())
print("Offset[]= ", [o.model()[i].as_long() for i in offsets])
print("Transitions:")
for (_, label), v in zip(TRANSITIONS, [o.model()[i].as_long() for i in trans_table]):
    print(f"{label:14} => {v:#10x}, // {v:032b}")

# Output should be deterministic:
# sat
# Offset[]=  [0, 6, 16, 19, 1, 25, 11, 18, 24]
# Transitions:
# 00-7F          =>      0x180, // 00000000000000000000000110000000
# C2-DF          =>      0x400, // 00000000000000000000010000000000
# E0             =>      0x4c0, // 00000000000000000000010011000000
# E1-EC, EE-EF   =>       0x40, // 00000000000000000000000001000000
# ED             =>      0x640, // 00000000000000000000011001000000
# F0             =>      0x2c0, // 00000000000000000000001011000000
# F1-F3          =>      0x480, // 00000000000000000000010010000000
# F4             =>      0x600, // 00000000000000000000011000000000
# 80-8F          => 0x21060020, // 00100001000001100000000000100000
# 90-9F          => 0x20060820, // 00100000000001100000100000100000
# A0-BF          =>   0x860820, // 00000000100001100000100000100000
# C0-C1, F5-FF   =>        0x0, // 00000000000000000000000000000000
