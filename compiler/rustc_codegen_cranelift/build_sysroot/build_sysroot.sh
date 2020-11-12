#!/bin/bash

# Requires the CHANNEL env var to be set to `debug` or `release.`

set -e

source ./config.sh

dir=$(pwd)

# Use rustc with cg_clif as hotpluggable backend instead of the custom cg_clif driver so that
# build scripts are still compiled using cg_llvm.
export RUSTC=$dir"/cg_clif_build_sysroot"
export RUSTFLAGS=$RUSTFLAGS" --clif"

cd $(dirname "$0")

# Cleanup for previous run
#     v Clean target dir except for build scripts and incremental cache
rm -r target/*/{debug,release}/{build,deps,examples,libsysroot*,native} 2>/dev/null || true

# We expect the target dir in the default location. Guard against the user changing it.
export CARGO_TARGET_DIR=target

# Build libs
export RUSTFLAGS="$RUSTFLAGS -Zforce-unstable-if-unmarked -Cpanic=abort"
if [[ "$1" != "--debug" ]]; then
    sysroot_channel='release'
    # FIXME Enable incremental again once rust-lang/rust#74946 is fixed
    # FIXME Enable -Zmir-opt-level=2 again once it doesn't ice anymore
    CARGO_INCREMENTAL=0 RUSTFLAGS="$RUSTFLAGS" cargo build --target $TARGET_TRIPLE --release
else
    sysroot_channel='debug'
    cargo build --target $TARGET_TRIPLE
fi

# Copy files to sysroot
mkdir -p $dir/sysroot/lib/rustlib/$TARGET_TRIPLE/lib/
cp -a target/$TARGET_TRIPLE/$sysroot_channel/deps/* $dir/sysroot/lib/rustlib/$TARGET_TRIPLE/lib/
