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

target_triple = sys.argv[14]

def normalize_path(v):
    """msys1/msys2 automatically converts `/abs/path1:/abs/path2` into
    `c:\real\abs\path1;c:\real\abs\path2` (semicolons) if shell thinks
    the value is list of paths.
    (if there is only one path, it becomes `c:/real/abs/path`.)
    this causes great confusion and error: shell and Makefile doesn't like
    windows paths so it is really error-prone. revert it for peace."""
    v = v.replace('\\', '/')
    # c:/path -> /c/path
    # "c:/path" -> "/c/path"
    start = v.find(':/')
    while start != -1:
        v = v[:start - 1] + '/' + v[start - 1:start] + v[start + 1:]
        start = v.find(':/')
    return v


def putenv(name, value):
    if os.name == 'nt':
        value = normalize_path(value)
    os.putenv(name, value)


def convert_path_spec(name, value):
    if os.name == 'nt' and name != 'PATH':
        value = ":".join(normalize_path(v) for v in value.split(";"))
    return value

make = sys.argv[2]
putenv('RUSTC', os.path.abspath(sys.argv[3]))
putenv('TMPDIR', os.path.abspath(sys.argv[4]))
putenv('CC', sys.argv[5] + ' ' + sys.argv[6])
putenv('RUSTDOC', os.path.abspath(sys.argv[7]))
filt = sys.argv[8]
putenv('LD_LIB_PATH_ENVVAR', sys.argv[9])
putenv('HOST_RPATH_DIR', os.path.abspath(sys.argv[10]))
putenv('TARGET_RPATH_DIR', os.path.abspath(sys.argv[11]))
putenv('RUST_BUILD_STAGE', sys.argv[12])
putenv('S', os.path.abspath(sys.argv[13]))
putenv('RUSTFLAGS', sys.argv[15])
putenv('LLVM_COMPONENTS', sys.argv[16])
putenv('PYTHON', sys.executable)
os.putenv('TARGET', target_triple)

if 'msvc' in target_triple:
    os.putenv('IS_MSVC', '1')

if filt not in sys.argv[1]:
    sys.exit(0)
print('maketest: ' + os.path.basename(os.path.dirname(sys.argv[1])))

path = sys.argv[1]
if path[-1] == '/':
    # msys1 has a bug that `make` fails to include `../tools.mk` (parent dir)
    # if `-C path` option is given and `path` is absolute directory with
    # trailing slash (`c:/path/to/test/`).
    # the easist workaround is to remove the slash (`c:/path/to/test`).
    # msys2 seems to fix this problem.
    path = path[:-1]

proc = subprocess.Popen([make, '-C', path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
out, err = proc.communicate()
i = proc.wait()

if i != 0:
    print """\
----- %s --------------------
------ stdout ---------------------------------------------
%s
------ stderr ---------------------------------------------
%s
------        ---------------------------------------------
""" % (sys.argv[1], out, err)

    sys.exit(i)
