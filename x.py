#!/usr/bin/env python3
# Some systems don't have `python3` in their PATH. This isn't supported by x.py directly;
# they should use `x` or `x.ps1` instead.

# This file is only a "symlink" to bootstrap.py, all logic should go there.

import os
import sys
import re
import logging

# If this is python2, check if python3 is available and re-execute with that
# interpreter. Only python3 allows downloading CI LLVM.
#
# This matters if someone's system `python` is python2.
if sys.version_info.major < 3:
    try:
        os.execvp("py", ["py", "-3"] + sys.argv)
    except OSError:
        try:
            os.execvp("python3", ["python3"] + sys.argv)
        except OSError:
            # Python 3 isn't available, fall back to python 2
            pass

rust_dir = os.path.dirname(os.path.abspath(__file__))

# Temporary 'fix' for https://github.com/rust-lang/rust/issues/56650.
# Various chunks of the build system can't correctly handle spaces in paths, and
# will break in unexpected ways if there are any.  This tests to see if there
# are spaces in the path, quitting with an error if there are any
if re.search("\s", rust_dir):
    logging.critical("There is a known bug in the build system "
                     "(https://github.com/rust-lang/rust/issues/56650) "
                     "that means that if there are spaces in your path then "
                     "the build system will fail.  Your path ('%s') contains "
                     "at least one space in it.  Either move your rust "
                     "repository to a path that has no spaces in it, or "
                     "change your path to remove all spaces.  Now quitting.",
                     rust_dir)
else:
    sys.path.append(os.path.join(rust_dir, "src", "bootstrap"))

    import bootstrap
    bootstrap.main()
