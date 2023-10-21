#!/usr/bin/env bash
set -e

./y.sh build

echo "[SETUP] Rust fork"
git clone https://github.com/rust-lang/rust.git || true
pushd rust
git fetch
git checkout -- .
git checkout "$(rustc -V | cut -d' ' -f3 | tr -d '(')"

git -c user.name=Dummy -c user.email=dummy@example.com -c commit.gpgSign=false \
    am ../patches/*-stdlib-*.patch

cat > config.toml <<EOF
change-id = 115898

[llvm]
ninja = false

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

export CFG_VIRTUAL_RUST_SOURCE_BASE_DIR="$(cd build/stdlib; pwd)"

# Allow the testsuite to use llvm tools
host_triple=$(rustc -vV | grep host | cut -d: -f2 | tr -d " ")
export LLVM_BIN_DIR="$(rustc --print sysroot)/lib/rustlib/$host_triple/bin"
