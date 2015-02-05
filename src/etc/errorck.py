# Copyright 2015 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# Digs error codes out of files named 'diagnostics.rs' across
# the tree, and ensures thare are no duplicates.

import sys
import os
import re

if len(sys.argv) < 2:
    print "usage: errorck.py <src-dir>"
    sys.exit(1)

src_dir = sys.argv[1]
errcode_map = {}
error_re = re.compile("(E\d\d\d\d)")

for (dirpath, dirnames, filenames) in os.walk(src_dir):
    if "src/test" in dirpath or "src/llvm" in dirpath:
        # Short circuit for fast
        continue

    for filename in filenames:
        if filename != "diagnostics.rs":
            continue

        path = os.path.join(dirpath, filename)

        with open(path, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                match = error_re.search(line)
                if match:
                    errcode = match.group(1)
                    new_record = [(errcode, path, line_num, line)]
                    existing = errcode_map.get(errcode)
                    if existing is not None:
                        # This is a dupe
                        errcode_map[errcode] = existing + new_record
                    else:
                        errcode_map[errcode] = new_record

errors = False
all_errors = []

for errcode, entries in errcode_map.items():
    all_errors.append(entries[0][0])
    if len(entries) > 1:
        print("error: duplicate error code " + errcode)
        for entry in entries:
            print("{1}: {2}\n{3}".format(*entry))
        errors = True

print
print("* {0} error codes".format(len(errcode_map)))
print("* highest error code: " + max(all_errors))
print

if errors:
    sys.exit(1)
