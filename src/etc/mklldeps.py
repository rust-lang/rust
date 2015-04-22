# Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import os
import sys
import subprocess

f = open(sys.argv[1], 'wb')

components = sys.argv[2].split(' ')
components = [i for i in components if i]  # ignore extra whitespaces
enable_static = sys.argv[3]
llconfig = sys.argv[4]

f.write("""// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// WARNING: THIS IS A GENERATED FILE, DO NOT MODIFY
//          take a look at src/etc/mklldeps.py if you're interested
""")


def run(args):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()

    if err:
        print("failed to run llconfig: args = `{}`".format(args))
        print(err)
        sys.exit(1)
    return out

f.write("\n")

# LLVM libs
args = [llconfig, '--libs', '--system-libs']

args.extend(components)
out = run(args)
for lib in out.strip().replace("\n", ' ').split(' '):
    if len(lib) == 0:
        continue
    # in some cases we get extra spaces in between libs so ignore those
    if len(lib) == 1 and lib == ' ':
        continue
    # not all libs strictly follow -lfoo, on Bitrig, there is -pthread
    if lib[0:2] == '-l':
        lib = lib.strip()[2:]
    elif lib[0] == '-':
        lib = lib.strip()[1:]
    f.write("#[link(name = \"" + lib + "\"")
    # LLVM libraries are all static libraries
    if 'LLVM' in lib:
        f.write(", kind = \"static\"")
    f.write(")]\n")

# LLVM ldflags
out = run([llconfig, '--ldflags'])
for lib in out.strip().split(' '):
    if lib[:2] == "-l":
        f.write("#[link(name = \"" + lib[2:] + "\")]\n")

# C++ runtime library
out = run([llconfig, '--cxxflags'])
if enable_static == '1':
    assert('stdlib=libc++' not in out)
    f.write("#[link(name = \"stdc++\", kind = \"static\")]\n")
else:
    if 'stdlib=libc++' in out:
        f.write("#[link(name = \"c++\")]\n")
    else:
        f.write("#[link(name = \"stdc++\")]\n")

# Attach everything to an extern block
f.write("extern {}\n")
