import re, os, sys, hashlib, tarfile, shutil, subprocess, tempfile

def scrub(b):
  if sys.version_info >= (3,) and type(b) == bytes:
    return b.decode('ascii')
  else:
    return b

src_dir = scrub(os.getenv("CFG_SRC_DIR"))
if not src_dir:
  raise Exception("missing env var CFG_SRC_DIR")

snapshotfile = os.path.join(src_dir, "src", "snapshots.txt")
download_url_base = "http://dl.rust-lang.org/stage0-snapshots"
download_dir_base = "dl"
download_unpack_base = os.path.join(download_dir_base, "unpack")

old_snapshot_files = {
    "linux": ["rustc", "lib/glue.o", "lib/libstd.so", "lib/libstd.rlib",
              "lib/librustrt.so", "lib/librustllvm.so"],
    "macos": ["rustc", "lib/glue.o", "lib/libstd.dylib", "lib/libstd.rlib",
              "lib/librustrt.dylib", "lib/librustllvm.dylib"],
    "winnt": ["rustc.exe", "lib/glue.o", "lib/std.dll", "lib/libstd.rlib",
              "lib/rustrt.dll", "lib/rustllvm.dll"]
    }

snapshot_files = {
    "linux": ["rustc", "lib/glue.o", "lib/libstd.so", "lib/libstd.rlib",
              "lib/librustrt.so", "lib/librustllvm.so", "intrinsics.bc"],
    "macos": ["rustc", "lib/glue.o", "lib/libstd.dylib", "lib/libstd.rlib",
              "lib/librustrt.dylib", "lib/librustllvm.dylib", "intrinsics.bc"],
    "winnt": ["rustc.exe", "lib/glue.o", "lib/std.dll", "lib/libstd.rlib",
              "lib/rustrt.dll", "lib/rustllvm.dll", "intrinsics.bc"]
    }

def parse_line(n, line):
  global snapshotfile

  if re.match(r"\s*$", line): return None

  if re.match(r"^T\s*$", line): return None

  match = re.match(r"\s+([\w_-]+) ([a-fA-F\d]{40})\s*$", line)
  if match:
    return { "type": "file",
             "platform": match.group(1),
             "hash": match.group(2).lower() }

  match = re.match(r"([ST]) (\d{4}-\d{2}-\d{2}) ([a-fA-F\d]+)\s*$", line);
  if (not match):
    raise Exception("%s:%d:E syntax error" % (snapshotfile, n))
  return {"type": "snapshot",
          "date": match.group(2),
          "rev": match.group(3)}


def partial_snapshot_name(date, rev, platform):
  return ("rust-stage0-%s-%s-%s.tar.bz2"
          % (date, rev, platform))

def full_snapshot_name(date, rev, platform, hsh):
  return ("rust-stage0-%s-%s-%s-%s.tar.bz2"
          % (date, rev, platform, hsh))


def get_kernel():
    if os.name == "nt" or scrub(os.getenv("CFG_ENABLE_MINGW_CROSS")):
        return "winnt"
    kernel = os.uname()[0].lower()
    if kernel == "darwin":
        kernel = "macos"
    return kernel


def get_cpu():
    # return os.uname()[-1].lower()
    return "i386"


def get_platform():
  return "%s-%s" % (get_kernel(), get_cpu())


def cmd_out(cmdline):
    p = subprocess.Popen(cmdline,
                         stdout=subprocess.PIPE)
    return scrub(p.communicate()[0].strip())


def local_rev_info(field):
    return cmd_out(["git", "--git-dir=" + os.path.join(src_dir, ".git"),
                    "log", "-n", "1",
                    "--format=%%%s" % field, "HEAD"])


def local_rev_full_sha():
    return local_rev_info("H").split()[0]


def local_rev_short_sha():
    return local_rev_info("h").split()[0]


def local_rev_committer_date():
    return local_rev_info("ci")

def get_url_to_file(u,f):
  subprocess.check_call(["curl", "-o", f, u])

def snap_filename_hash_part(snap):
  match = re.match(r".*([a-fA-F\d]{40}).tar.bz2$", snap)
  if not match:
    raise Exception("unable to find hash in filename: " + snap)
  return match.group(1)

def hash_file(x):
    h = hashlib.sha1()
    h.update(open(x, "rb").read())
    return scrub(h.hexdigest())


def make_snapshot(stage):
    kernel = get_kernel()
    platform = get_platform()
    rev = local_rev_short_sha()
    date = local_rev_committer_date().split()[0]

    file0 = partial_snapshot_name(date, rev, platform)

    tar = tarfile.open(file0, "w:bz2")
    for name in snapshot_files[kernel]:
      tar.add(os.path.join(stage, name),
              "rust-stage0/" + name)
    tar.close()

    h = hash_file(file0)
    file1 = full_snapshot_name(date, rev, platform, h)

    shutil.move(file0, file1)
    return file1
