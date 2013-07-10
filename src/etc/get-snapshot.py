#!/usr/bin/env python
# xfail-license

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
    shutil.move(tp, fp)
  tar.close()
  shutil.rmtree(download_unpack_base)

def determine_curr_snapshot(triple):
  i = 0
  platform = get_platform(triple)

  found_file = False
  found_snap = False
  hsh = None
  date = None
  rev = None

  f = open(snapshotfile)
  for line in f.readlines():
    i += 1
    parsed = parse_line(i, line)
    if (not parsed): continue

    if found_snap and parsed["type"] == "file":
      if parsed["platform"] == platform:
        hsh = parsed["hash"]
        found_file = True
        break;
    elif parsed["type"] == "snapshot":
      date = parsed["date"]
      rev = parsed["rev"]
      found_snap = True

  if not found_snap:
    raise Exception("no snapshot entries in file")

  if not found_file:
    raise Exception("no snapshot file found for platform %s, rev %s" %
                    (platform, rev))

  return full_snapshot_name(date, rev, platform, hsh)

# Main

# this gets called with one or two arguments:
# The first is the O/S triple.
# The second is an optional path to the snapshot to use.

triple = sys.argv[1]
if len(sys.argv) == 3:
  dl_path = sys.argv[2]
else:
  snap = determine_curr_snapshot(triple)
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
