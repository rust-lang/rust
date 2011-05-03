#!/usr/bin/env python

import shutil, tarfile
from snapshot import *

kernel = get_kernel()
cpu = get_cpu()
rev = local_rev_short_sha()
date = local_rev_committer_date().split()[0]

file0 = partial_snapshot_name(date, rev, kernel, cpu)

tar = tarfile.open(file0, "w:bz2")
for name in snapshot_files[kernel]:
    tar.add(os.path.join("stage2", name),
            os.path.join("rust-stage0", name))
tar.close()

h = hash_file(file0)
file1 = full_snapshot_name(date, rev, kernel, cpu, h)

shutil.move(file0, file1)

print(file1)
