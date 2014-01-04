#!/usr/bin/env python
# xfail-license

# Copies Rust runtime dependencies to the specified directory

import snapshot, sys, os, shutil

def copy_runtime_deps(dest_dir):
    for path in snapshot.get_winnt_runtime_deps():
        shutil.copy(path, dest_dir)

    lic_dest = os.path.join(dest_dir, "third-party")
    if os.path.exists(lic_dest):
        shutil.rmtree(lic_dest) # copytree() won't overwrite existing files
    shutil.copytree(os.path.join(os.path.dirname(__file__), "third-party"), lic_dest)

copy_runtime_deps(sys.argv[1])
