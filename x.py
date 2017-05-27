#!/usr/bin/env python
# Copyright 2016 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# This file is only a "symlink" to bootstrap.py, all logic should go there.

import os
import sys
rust_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(rust_dir, "src", "bootstrap"))

import bootstrap
bootstrap.main()
