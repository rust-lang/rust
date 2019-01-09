#!/usr/bin/env python

# This file is only a "symlink" to bootstrap.py, all logic should go there.

import os
import sys
rust_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(rust_dir, "src", "bootstrap"))

import bootstrap
bootstrap.main()
