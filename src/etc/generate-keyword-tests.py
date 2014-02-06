#!/usr/bin/env python
#
# Copyright 2013 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.
"""
This script takes a list of keywords and generates a testcase, that checks
if using the keyword as identifier fails, for every keyword. The generate
test files are set read-only.
Test for https://github.com/mozilla/rust/issues/2275

sample usage: src/etc/generate-keyword-tests.py as break
"""

import sys
import os
import datetime
import stat


template = """// Copyright %d The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This file was auto-generated using 'src/etc/generate-keyword-tests.py %s'

fn main() {
    let %s = "foo"; //~ error: ident
}
"""

test_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../test/compile-fail')
)

for kw in sys.argv[1:]:
    test_file = os.path.join(test_dir, 'keyword-%s-as-identifier.rs' % kw)

    # set write permission if file exists, so it can be changed
    if os.path.exists(test_file):
        os.chmod(test_file, stat.S_IWUSR)

    with open(test_file, 'wt') as f:
        f.write(template % (datetime.datetime.now().year, kw, kw))

    # mark file read-only
    os.chmod(test_file, stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH)
