#!/usr/bin/env python

import os, tarfile, hashlib, re, shutil
from snapshot import *

def snap_filename_hash_part(snap):
  match = re.match(r".*([a-fA-F\d]{40}).tar.bz2$", snap)
  if not match:
    raise Exception("unable to find hash in filename: " + snap)
  return match.group(1)

def get_snapshot_and_check_hash(snap):

  hsh = snap_filename_hash_part(snap)

  h = hashlib.sha1()
  url = download_url_base + "/" + snap
  print "downloading " + url
  u = urllib2.urlopen(url)
  print "checking hash on download"
  data = u.read()
  h.update(data)
  if h.hexdigest() != hsh:
    raise Exception("hash check failed on " + snap)

  print "hash ok"
  with open(os.path.join(download_dir_base, snap), "w+b") as f:
    f.write(data)
  return True

def unpack_snapshot(snap):
  dl_path = os.path.join(download_dir_base, snap)
  print "opening snapshot " + dl_path
  tar = tarfile.open(dl_path)
  kernel = get_kernel()
  for name in snapshot_files[kernel]:
    p = os.path.join("rust-stage0", name)
    fp = os.path.join("stage0", name)
    print "extracting " + fp
    tar.extract(p, download_unpack_base)
    tp = os.path.join(download_unpack_base, p)
    shutil.move(tp, fp)
  tar.close()
  shutil.rmtree(download_unpack_base)

def determine_last_snapshot_for_platform():
  lines = open(snapshotfile).readlines();

  platform = get_platform()

  found = False
  hsh = None
  date = None
  rev = None

  for ln in range(len(lines) - 1, -1, -1):
    parsed = parse_line(ln, lines[ln])
    if (not parsed): continue

    if parsed["type"] == "file":
      if parsed["platform"] == platform:
        hsh = parsed["hash"]
    elif parsed["type"] == "snapshot":
      date = parsed["date"]
      rev = parsed["rev"]
      found = True
      break
    elif parsed["type"] == "transition" and not foundSnapshot:
      raise Exception("working on a transition, not updating stage0")

  if not found:
    raise Exception("no snapshot entries in file")

  if not hsh:
    raise Exception("no snapshot file found for platform %s, rev %s" %
                    (platform, rev))

  return full_snapshot_name(date, rev, get_kernel(), get_cpu(), hsh)

# Main

snap = determine_last_snapshot_for_platform()
print "determined most recent snapshot: " + snap
dl = os.path.join(download_dir_base, snap)
if (os.path.exists(dl)):
  if (snap_filename_hash_part(snap) == hash_file(dl)):
    print "found existing download with ok hash"
  else:
    print "bad hash on existing download, re-fetching"
    get_snapshot_and_check_hash(snap)
else:
  print "no cached download, fetching"
  get_snapshot_and_check_hash(snap)

unpack_snapshot(snap)
