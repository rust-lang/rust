# ignore-tidy-linelength
import os
import subprocess
from pathlib import Path
from urllib.request import urlretrieve
import tarfile


artifact_url = "https://ci-artifacts.rust-lang.org/rustc-builds/de1bc0008be096cf7ed67b93402250d3b3e480d0/reproducible-artifacts-nightly-x86_64-unknown-linux-gnu.tar.xz"

archive = "artifacts.tar.xz"
urlretrieve(artifact_url, archive)

with tarfile.open(archive) as f:
    f.extractall("artifacts")

llvm_pgo_path = Path("artifacts/reproducible-artifacts-nightly-x86_64-unknown-linux-gnu/reproducible-artifacts/llvm-pgo.profdata")
assert llvm_pgo_path.exists()

env = os.environ.copy()
env["RUST_BACKTRACE"] = "full"
subprocess.run([
    "python", "x.py", "dist",
    "--llvm-profile-use", str(llvm_pgo_path)
], check=True, env=env)
