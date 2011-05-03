#!/usr/bin/env python

import os, tarfile, hashlib, re, shutil
from snapshot import *

def snap_filename_hash_part(snap):
  match = re.match(r".*([a-fA-F\d]{40}).tar.bz2$", snap)
  if not match:
    raise Exception("unable to find hash in filename: " + snap)
  return match.group(1)

def unpack_snapshot(snap):
  dl_path = os.path.join(download_dir_base, snap)
  print("opening snapshot " + dl_path)
  tar = tarfile.open(dl_path)
  kernel = get_kernel()
  for name in snapshot_files[kernel]:
    p = os.path.join("rust-stage0", name)
    fp = os.path.join("stage0", name)
    print("extracting " + fp)
    tar.extract(p, download_unpack_base)
    tp = os.path.join(download_unpack_base, p)
    shutil.move(tp, fp)
  tar.close()
  shutil.rmtree(download_unpack_base)

def determine_curr_snapshot_for_platform():

  i = 0
  platform = get_platform()

  found_file = False
  found_snap = False
  hsh = None
  date = None
  rev = None

  with open(snapshotfile) as f:
    for line in f.xreadlines():
      i += 1
      parsed = parse_line(i, line)
      if (not parsed): continue

      if parsed["type"] == "transition":
        raise Exception("working on a transition, not updating stage0")

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

  return full_snapshot_name(date, rev, get_platform(), hsh)

# Main

snap = determine_curr_snapshot_for_platform()
dl = os.path.join(download_dir_base, snap)
url = download_url_base + "/" + snap
print("determined most recent snapshot: " + snap)

if (not os.path.exists(dl)):
  get_url_to_file(url, dl)

if (snap_filename_hash_part(snap) == hash_file(dl)):
  print("got download with ok hash")
else:
  raise Exception("bad hash on download")

unpack_snapshot(snap)
