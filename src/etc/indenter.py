#!/usr/bin/env python
from __future__ import print_function

import re
import sys

indent = 0
more_re = re.compile(r"^rust: ~\">>")
less_re = re.compile(r"^rust: ~\"<<")
while True:
    line = sys.stdin.readline()
    if not line:
        break

    if more_re.match(line):
        indent += 1

    print("%03d %s%s" % (indent, " " * indent, line.strip()))

    if less_re.match(line):
        indent -= 1
