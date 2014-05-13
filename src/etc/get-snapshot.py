#!/usr/bin/env python
#
# Copyright 2011-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import os, tarfile, re, shutil, sys
from snapshot import *

def unpack_snapshot(triple, dl_path):
  print("opening snapshot " + dl_path)
  tar = tarfile.open(dl_path)
  kernel = get_kernel(triple)

  stagep = os.path.join(triple, "stage0")

  # Remove files from prior unpackings, since snapshot rustc may not
  # be able to disambiguate between multiple candidate libraries.
  # (Leave dirs in place since extracting step still needs them.)
  for root, _, files in os.walk(stagep):
    for f in files:
      print("removing " + os.path.join(root, f))
      os.unlink(os.path.join(root, f))

  for p in tar.getnames():
    name = p.replace("rust-stage0/", "", 1);

    fp = os.path.join(stagep, name)
    print("extracting " + p)
    tar.extract(p, download_unpack_base)
    tp = os.path.join(download_unpack_base, p)
    if os.path.isdir(tp) and os.path.exists(fp):
        continue
    shutil.move(tp, fp)
  tar.close()
  shutil.rmtree(download_unpack_base)


# Main

# this gets called with one or two arguments:
# The first is the O/S triple.
# The second is an optional path to the snapshot to use.

triple = sys.argv[1]
if len(sys.argv) == 3:
  dl_path = sys.argv[2]
else:
  # There are no 64-bit Windows snapshots yet, so we'll use 32-bit ones instead, for now
  snap_triple = triple if triple != "x86_64-w64-mingw32" else "i686-pc-mingw32"
  snap = determine_curr_snapshot(snap_triple)
  dl = os.path.join(download_dir_base, snap)
  url = download_url_base + "/" + snap
  print("determined most recent snapshot: " + snap)

  if (not os.path.exists(dl)):
    get_url_to_file(url, dl)

  if (snap_filename_hash_part(snap) == hash_file(dl)):
    print("got download with ok hash")
  else:
    raise Exception("bad hash on download")

  dl_path = os.path.join(download_dir_base, snap)

unpack_snapshot(triple, dl_path)
