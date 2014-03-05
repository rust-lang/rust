# Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import subprocess
import os
import sys

os.putenv('RUSTC', os.path.abspath(sys.argv[2]))
os.putenv('TMPDIR', os.path.abspath(sys.argv[3]))
os.putenv('CC', sys.argv[4])
os.putenv('RUSTDOC', os.path.abspath(sys.argv[5]))
filt = sys.argv[6]
ldpath = sys.argv[7]
if ldpath != '':
    os.putenv(ldpath.split('=')[0], ldpath.split('=')[1])

if not filt in sys.argv[1]:
    sys.exit(0)
print('maketest: ' + os.path.basename(os.path.dirname(sys.argv[1])))

proc = subprocess.Popen(['make', '-C', sys.argv[1]],
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE)
out, err = proc.communicate()
i = proc.wait()

if i != 0:

    print '----- ' + sys.argv[1] + """ --------------------
------ stdout ---------------------------------------------
""" + out + """
------ stderr ---------------------------------------------
""" + err + """
------        ---------------------------------------------
"""
    sys.exit(i)

