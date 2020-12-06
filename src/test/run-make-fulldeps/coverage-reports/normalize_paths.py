#!/usr/bin/env python

import sys

# Normalize file paths in output
for line in sys.stdin:
    if line.startswith("..") and line.rstrip().endswith(".rs:"):
        print(line.replace("\\", "/"), end='')
    else:
        print(line, end='')
