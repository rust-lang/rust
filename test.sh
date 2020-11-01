#!/bin/bash
set -e

# Build cg_clif
export RUSTFLAGS="-Zrun_dsymutil=no"
if [[ "$1" == "--release" ]]; then
    export CHANNEL='release'
    cargo build --release
else
    export CHANNEL='debug'
    cargo build --bin cg_clif
fi

# Config
source scripts/config.sh
source scripts/tests.sh
export CG_CLIF_INCR_CACHE_DISABLED=1

# Cleanup
rm -r target/out || true

no_sysroot_tests

echo "[BUILD] sysroot"
time ./build_sysroot/build_sysroot.sh --release

base_sysroot_tests

extended_sysroot_tests
