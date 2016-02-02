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
    print("usage: errorck.py <src-dir>")
    sys.exit(1)

src_dir = sys.argv[1]
errcode_map = {}
errcode_checked = []
errcode_not_found = []
error_re = re.compile("(E\d\d\d\d)")

def check_unused_error_codes(error_codes, check_error_codes, filenames, dirnames, dirpath):
    for filename in filenames:
        if filename == "diagnostics.rs" or not filename.endswith(".rs"):
            continue
        path = os.path.join(dirpath, filename)

        with open(path, 'r') as f:
            for line in f:
                match = error_re.search(line)
                if match:
                    errcode = match.group(1)
                    if errcode in error_codes:
                        error_codes.remove(errcode)
                    if errcode not in check_error_codes:
                        check_error_codes.append(errcode)
    for dirname in dirnames:
        path = os.path.join(dirpath, dirname)
        for (dirpath, dnames, fnames) in os.walk(path):
            check_unused_error_codes(error_codes, check_error_codes, fnames, dnames, dirpath)


# In the register_long_diagnostics! macro, entries look like this:
#
# EXXXX: r##"
# <Long diagnostic message>
# "##,
#
# These two variables are for detecting the beginning and end of diagnostic
# messages so that duplicate error codes are not reported when a code occurs
# inside a diagnostic message
long_diag_begin = "r##\""
long_diag_end = "\"##"

errors = False
all_errors = []

for (dirpath, dirnames, filenames) in os.walk(src_dir):
    if "src/test" in dirpath or "src/llvm" in dirpath:
        # Short circuit for fast
        continue

    errcode_to_check = []
    for filename in filenames:
        if filename != "diagnostics.rs":
            continue
        path = os.path.join(dirpath, filename)

        with open(path, 'r') as f:
            inside_long_diag = False
            errcode_to_check = []
            for line_num, line in enumerate(f, start=1):
                if inside_long_diag:
                    # Skip duplicate error code checking for this line
                    if long_diag_end in line:
                        inside_long_diag = False
                    continue

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
                        # we don't check if this is a long error explanation
                        if (long_diag_begin not in line and not line.strip().startswith("//")
                            and errcode not in errcode_to_check and errcode not in errcode_checked
                            and errcode not in errcode_not_found):
                            errcode_to_check.append(errcode)

                if long_diag_begin in line:
                    inside_long_diag = True
        break
    check_unused_error_codes(errcode_to_check, errcode_checked, filenames, dirnames, dirpath)
    if len(errcode_to_check) > 0:
        for errcode in errcode_to_check:
            if errcode in errcode_checked:
                continue
            errcode_not_found.append(errcode)

if len(errcode_not_found) > 0:
    errcode_not_found.sort()
    for errcode in errcode_not_found:
        if errcode in errcode_checked:
            continue
        all_errors.append(errcode)
        print("error: unused error code: {0} ({1}:{2})".format(*errcode_map[errcode][0]))
        errors = True


for errcode, entries in errcode_map.items():
    all_errors.append(entries[0][0])
    if len(entries) > 1:
        entries.sort()
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
