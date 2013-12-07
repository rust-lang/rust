# xfail-license

import re, os, sys, glob, tarfile, shutil, subprocess, tempfile, distutils.spawn

try:
  import hashlib
  sha_func = hashlib.sha1
except ImportError:
  import sha
  sha_func = sha.new

def scrub(b):
  if sys.version_info >= (3,) and type(b) == bytes:
    return b.decode('ascii')
  else:
    return b

src_dir = scrub(os.getenv("CFG_SRC_DIR"))
if not src_dir:
  raise Exception("missing env var CFG_SRC_DIR")

snapshotfile = os.path.join(src_dir, "src", "snapshots.txt")
download_url_base = "http://static.rust-lang.org/stage0-snapshots"
download_dir_base = "dl"
download_unpack_base = os.path.join(download_dir_base, "unpack")

snapshot_files = {
    "linux": ["bin/rustc"],
    "macos": ["bin/rustc"],
    "winnt": ["bin/rustc.exe"],
    "freebsd": ["bin/rustc"],
    }

winnt_runtime_deps = ["libgcc_s_dw2-1.dll",
                      "libstdc++-6.dll",
                      "libpthread-2.dll"]

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
    raise Exception("%s:%d:E syntax error: " % (snapshotfile, n))
  return {"type": "snapshot",
          "date": match.group(2),
          "rev": match.group(3)}


def partial_snapshot_name(date, rev, platform):
  return ("rust-stage0-%s-%s-%s.tar.bz2"
          % (date, rev, platform))

def full_snapshot_name(date, rev, platform, hsh):
  return ("rust-stage0-%s-%s-%s-%s.tar.bz2"
          % (date, rev, platform, hsh))


def get_kernel(triple):
    os_name = triple.split('-')[-1]
    #scrub(os.getenv("CFG_ENABLE_MINGW_CROSS")):
    if os_name == "nt" or os_name == "mingw32":
        return "winnt"
    if os_name == "darwin":
        return "macos"
    if os_name == "freebsd":
        return "freebsd"
    return "linux"

def get_cpu(triple):
    arch = triple.split('-')[0]
    if arch == "i686":
      return "i386"
    return arch

def get_platform(triple):
  return "%s-%s" % (get_kernel(triple), get_cpu(triple))


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
    # no security issue, just to stop partial download leaving a stale file
    tmpf = f + '.tmp'

    returncode = -1
    if distutils.spawn.find_executable("curl"):
        returncode = subprocess.call(["curl", "-o", tmpf, u])
    elif distutils.spawn.find_executable("wget"):
        returncode = subprocess.call(["wget", "-O", tmpf, u])

    if returncode != 0:
        try:
            os.unlink(tmpf)
        except OSError as e:
            pass
        raise Exception("failed to fetch url")
    os.rename(tmpf, f)

def snap_filename_hash_part(snap):
  match = re.match(r".*([a-fA-F\d]{40}).tar.bz2$", snap)
  if not match:
    raise Exception("unable to find hash in filename: " + snap)
  return match.group(1)

def hash_file(x):
    h = sha_func()
    h.update(open(x, "rb").read())
    return scrub(h.hexdigest())

# Returns a list of paths of Rust's system runtime dependencies
def get_winnt_runtime_deps():
    runtime_deps = []
    path_dirs = os.environ["PATH"].split(';')
    for name in winnt_runtime_deps:
      for dir in path_dirs:
        matches = glob.glob(os.path.join(dir, name))
        if matches:
          runtime_deps.append(matches[0])
          break
      else:
        raise Exception("Could not find runtime dependency: %s" % name)
    return runtime_deps

def make_snapshot(stage, triple):
    kernel = get_kernel(triple)
    platform = get_platform(triple)
    rev = local_rev_short_sha()
    date = local_rev_committer_date().split()[0]

    file0 = partial_snapshot_name(date, rev, platform)

    def in_tar_name(fn):
      cs = re.split(r"[\\/]", fn)
      if len(cs) >= 2:
        return os.sep.join(cs[-2:])

    tar = tarfile.open(file0, "w:bz2")

    for name in snapshot_files[kernel]:
      dir = stage
      if stage == "stage1" and re.match(r"^lib/(lib)?std.*", name):
        dir = "stage0"
      fn_glob = os.path.join(triple, dir, name)
      matches = glob.glob(fn_glob)
      if not matches:
        raise Exception("Not found file with name like " + fn_glob)
      if len(matches) == 1:
        tar.add(matches[0], "rust-stage0/" + in_tar_name(matches[0]))
      else:
        raise Exception("Found stale files: \n  %s\n"
                        "Please make a clean build." % "\n  ".join(matches))

    if kernel=="winnt":
      for path in get_winnt_runtime_deps():
        tar.add(path, "rust-stage0/bin/" + os.path.basename(path))
      tar.add(os.path.join(os.path.dirname(__file__), "third-party"),
              "rust-stage0/bin/third-party")

    tar.close()

    h = hash_file(file0)
    file1 = full_snapshot_name(date, rev, platform, h)

    shutil.move(file0, file1)

    return file1
