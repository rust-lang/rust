#!/bin/bash
set -e
cd $(dirname "$0")

# Cleanup for previous run
#     v Clean target dir except for build scripts and incremental cache
rm -r target/*/{debug,release}/{build,deps,examples,libsysroot*,native} || true
rm Cargo.lock 2>/dev/null || true
rm -r sysroot 2>/dev/null || true

# FIXME find a better way to get the target triple
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   TARGET_TRIPLE='x86_64-unknown-linux-gnu'
elif [[ "$unamestr" == 'Darwin' ]]; then
   TARGET_TRIPLE='x86_64-apple-darwin'
else
   echo "Unsupported os"
   exit 1
fi

# Build libs
mkdir -p sysroot/lib/rustlib/$TARGET_TRIPLE/lib/
export RUSTFLAGS="$RUSTFLAGS -Z force-unstable-if-unmarked"
if [[ "$1" == "--release" ]]; then
    channel='release'
    RUSTFLAGS="$RUSTFLAGS -Zmir-opt-level=3" cargo build --target $TARGET_TRIPLE --release
else
    channel='debug'
    cargo build --target $TARGET_TRIPLE
fi

# Copy files to sysroot
cp target/$TARGET_TRIPLE/$channel/deps/*.rlib sysroot/lib/rustlib/$TARGET_TRIPLE/lib/
