#!/usr/bin/env python

"""
This script takes a list of keywords and generates a testcase, that checks
if using the keyword as identifier fails, for every keyword. The generate
test files are set read-only.
Test for https://github.com/rust-lang/rust/issues/2275

sample usage: src/etc/generate-keyword-tests.py as break
"""

import sys
import os
import stat


template = """\
// This file was auto-generated using 'src/etc/generate-keyword-tests.py %s'

fn main() {
    let %s = "foo"; //~ error: expected pattern, found keyword `%s`
}
"""

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../test/ui/parser"))

for kw in sys.argv[1:]:
    test_file = os.path.join(test_dir, "keyword-%s-as-identifier.rs" % kw)

    # set write permission if file exists, so it can be changed
    if os.path.exists(test_file):
        os.chmod(test_file, stat.S_IWUSR)

    with open(test_file, "wt") as f:
        f.write(template % (kw, kw, kw))

    # mark file read-only
    os.chmod(test_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
