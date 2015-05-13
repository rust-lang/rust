#!/usr/bin/env python
#
# Copyright 2015 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# This script takes n+1 arguments. The first argument is a path to the
# serialized error metadata file (usually a json file, though that
# does not actually matter here). The rest of the arguments are the
# paths to source files that we generate the error metadata file.
#
# If any of the source files are newer than the error metadata file,
# then we *delete* the error metadata file. The driving assumption is
# that it will be regenerated during the subsequent make-driven build.
#
# It is not an error if any of the files do not exist; they are simply
# left out of the comparison in that case.

import os
import sys

if __name__ == '__main__':
    metadata = sys.argv[1]
    # print "Running %s on metadata %s" % (sys.argv[0], metadata)
    if not os.path.exists(metadata):
        # print "Skipping metadata %s; does not exist" % metadata
        sys.exit(0)
    metadata_mtime = os.path.getmtime(metadata);
    source_files = sys.argv[2:]
    for f in source_files:
        if not os.path.exists(f):
            # print "Skipping comparison with %s since latter does not exist" % f
            continue
        f_mtime = os.path.getmtime(f);
        # print("Comparing %s against %s" % (f, metadata))
        # print("time (%d) against time (%d)" % (f_mtime, metadata_mtime))
        if f_mtime > metadata_mtime:
            print "Removing %s since %s is newer" % (metadata, f)
            os.remove(metadata)
        sys.exit(0)
