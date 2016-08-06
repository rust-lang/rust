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

components = sys.argv[2].split() # splits on whitespace
enable_static = sys.argv[3]
llvm_config = sys.argv[4]
stdcpp_name = sys.argv[5]
use_libcpp = sys.argv[6]

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
        print("failed to run llvm_config: args = `{}`".format(args))
        print(err)
        sys.exit(1)
    return out

def runErr(args):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()

    if err:
        return False, out
    else:
        return True, out

f.write("\n")

args = [llvm_config, '--shared-mode']
args.extend(components)
llvm_shared, out = runErr(args)
if llvm_shared:
    llvm_shared = 'shared' in out

# LLVM libs
args = [llvm_config, '--libs', '--system-libs']
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
    # If this actually points at a literal file then we're on MSVC which now
    # prints full paths, so get just the name of the library and strip off the
    # trailing ".lib"
    elif os.path.exists(lib):
        lib = os.path.basename(lib)[:-4]
    elif lib[-4:] == '.lib':
        lib = lib[:-4]
    f.write("#[link(name = \"" + lib + "\"")
    if not llvm_shared and 'LLVM' in lib:
        f.write(", kind = \"static\"")
    f.write(")]\n")

# LLVM ldflags
out = run([llvm_config, '--ldflags'])
for lib in out.strip().split(' '):
    if lib[:2] == "-l":
        f.write("#[link(name = \"" + lib[2:] + "\")]\n")

# C++ runtime library
out = run([llvm_config, '--cxxflags'])
if enable_static == '1':
    assert('stdlib=libc++' not in out)
    f.write("#[link(name = \"" + stdcpp_name + "\", kind = \"static\")]\n")
else:
    # Note that we use `cfg_attr` here because on MSVC the C++ standard library
    # is not c++ or stdc++, but rather the linker takes care of linking the
    # right standard library.
    if use_libcpp != "0" or 'stdlib=libc++' in out:
        f.write("#[cfg_attr(not(target_env = \"msvc\"), link(name = \"c++\"))]\n")
    else:
        f.write("#[cfg_attr(not(target_env = \"msvc\"), link(name = \"" + stdcpp_name + "\"))]\n")

# Attach everything to an extern block
f.write("extern {}\n")
