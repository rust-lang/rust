# Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# Copies Rust runtime dependencies to the specified directory.

import snapshot, sys, os, shutil

def copy_runtime_deps(dest_dir):
    for path in snapshot.get_winnt_runtime_deps():
        shutil.copy(path, dest_dir)

    lic_dest = os.path.join(dest_dir, "third-party")
    if os.path.exists(lic_dest):
        shutil.rmtree(lic_dest) # copytree() won't overwrite existing files
    shutil.copytree(os.path.join(os.path.dirname(__file__), "third-party"), lic_dest)

copy_runtime_deps(sys.argv[1])
