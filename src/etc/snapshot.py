import re, os, sys, hashlib, tarfile, shutil, subprocess, urllib2, tempfile

snapshotfile = "snapshots.txt"
download_url_base = "http://dl.rust-lang.org/stage0-snapshots"
download_dir_base = "dl"
download_unpack_base = os.path.join(download_dir_base, "unpack")

snapshot_files = {
    "linux": ["rustc", "glue.o", "libstd.so" ],
    "macos": ["rustc", "glue.o", "libstd.dylib" ],
    "winnt": ["rustc.exe", "glue.o", "std.dll" ]
    }

def parse_line(n, line):
  global snapshotfile

  if re.match(r"\s*$", line): return None

  match = re.match(r"\s+([\w_-]+) ([a-fA-F\d]{40})\s*$", line)
  if match:
    return { "type": "file",
             "platform": match.group(1),
             "hash": match.group(2).lower() }

  match = re.match(r"([ST]) (\d{4}-\d{2}-\d{2}) ([a-fA-F\d]+)\s*$", line);
  if (not match):
    raise Exception("%s:%d:E syntax error" % (snapshotfile, n))
  ttype = "snapshot"
  if (match.group(1) == "T"):
    ttype = "transition"
  return {"type": ttype,
          "date": match.group(2),
          "rev": match.group(3)}


def partial_snapshot_name(date, rev, kernel, cpu):
  return ("rust-stage0-%s-%s-%s-%s.tar.bz2"
          % (date, rev, kernel, cpu))

def full_snapshot_name(date, rev, kernel, cpu, hsh):
  return ("rust-stage0-%s-%s-%s-%s-%s.tar.bz2"
          % (date, rev, kernel, cpu, hsh))


def get_kernel():
    if os.name == "nt":
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
    return p.communicate()[0].strip()


def local_rev_info(field):
    return cmd_out(["git", "log", "-n", "1",
                    "--format=%%%s" % field, "HEAD"])


def local_rev_full_sha():
    return local_rev_info("H").split()[0]


def local_rev_short_sha():
    return local_rev_info("h").split()[0]


def local_rev_committer_date():
    return local_rev_info("ci")


def hash_file(x):
    h = hashlib.sha1()
    h.update(open(x).read())
    return h.hexdigest()
