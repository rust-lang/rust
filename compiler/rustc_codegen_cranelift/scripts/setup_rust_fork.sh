#!/usr/bin/env bash
set -e

# CG_CLIF_FORCE_GNU_AS will force usage of as instead of the LLVM backend of rustc as
# the LLVM backend isn't compiled in here.
export CG_CLIF_FORCE_GNU_AS=1

# Compiletest expects all standard library paths to start with /rustc/FAKE_PREFIX.
# CG_CLIF_STDLIB_REMAP_PATH_PREFIX will cause cg_clif's build system to pass
# --remap-path-prefix to handle this.
CG_CLIF_STDLIB_REMAP_PATH_PREFIX=/rustc/FAKE_PREFIX ./y.sh build

echo "[SETUP] Rust fork"
git clone --quiet https://github.com/rust-lang/rust.git --filter=tree:0 || true
pushd rust
git fetch
git checkout --no-progress -- .
git checkout --no-progress "$(rustc -V | cut -d' ' -f3 | tr -d '(')"

git submodule update --quiet --init src/tools/cargo library/backtrace library/stdarch

git -c user.name=Dummy -c user.email=dummy@example.com -c commit.gpgSign=false \
    am ../patches/*-stdlib-*.patch

cat > config.toml <<EOF
change-id = 999999

[llvm]
download-ci-llvm = true

[build]
rustc = "$(pwd)/../dist/bin/rustc-clif"
cargo = "$(rustup which cargo)"
full-bootstrap = true
local-rebuild = true

[rust]
codegen-backends = ["cranelift"]
deny-warnings = false
verbose-tests = false
EOF
popd

# Allow the testsuite to use llvm tools
host_triple=$(rustc -vV | grep host | cut -d: -f2 | tr -d " ")
export LLVM_BIN_DIR="$(rustc --print sysroot)/lib/rustlib/$host_triple/bin"
