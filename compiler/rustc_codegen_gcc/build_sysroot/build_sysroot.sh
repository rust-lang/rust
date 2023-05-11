#!/usr/bin/env bash

# Requires the CHANNEL env var to be set to `debug` or `release.`

set -e
cd $(dirname "$0")

pushd ../ >/dev/null
source ./config.sh
popd >/dev/null

# Cleanup for previous run
#     v Clean target dir except for build scripts and incremental cache
rm -r target/*/{debug,release}/{build,deps,examples,libsysroot*,native} 2>/dev/null || true
rm Cargo.lock test_target/Cargo.lock 2>/dev/null || true
rm -r sysroot/ 2>/dev/null || true

# Build libs
export RUSTFLAGS="$RUSTFLAGS -Z force-unstable-if-unmarked"
if [[ "$1" == "--release" ]]; then
    sysroot_channel='release'
    RUSTFLAGS="$RUSTFLAGS -Zmir-opt-level=3" cargo build --target $TARGET_TRIPLE --release
else
    sysroot_channel='debug'
    cargo build --target $TARGET_TRIPLE --features compiler_builtins/c
fi

# Copy files to sysroot
mkdir -p sysroot/lib/rustlib/$TARGET_TRIPLE/lib/
cp -r target/$TARGET_TRIPLE/$sysroot_channel/deps/* sysroot/lib/rustlib/$TARGET_TRIPLE/lib/
