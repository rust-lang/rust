#!/usr/bin/env python
# A simple wrapper that forwards the arguments to bash, unless the
# CI_OVERRIDE_SHELL environment variable is present: in that case the content
# of that environment variable is used as the shell path.

import os
import sys
import subprocess

try:
    shell = os.environ["CI_OVERRIDE_SHELL"]
except KeyError:
    shell = "bash"

res = subprocess.call([shell] + sys.argv[1:])
sys.exit(res)
