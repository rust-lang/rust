#!/bin/sh
set -exuo pipefail

CRATE=example

env | sort
mkdir -p $WORK_DIR
pushd $WORK_DIR
    rm -rf $CRATE || echo OK
    cp -a $HERE/example .
    pushd $CRATE
        # HACK(eddyb) sets `RUSTC_BOOTSTRAP=1` so Cargo can accept nightly features.
        # These come from the top-level Rust workspace, that this crate is not a
        # member of, but Cargo tries to load the workspace `Cargo.toml` anyway.
        env RUSTC_BOOTSTRAP=1 RUSTFLAGS="-C linker=arm-none-eabi-ld -C link-arg=-Tlink.x" \
            $BOOTSTRAP_CARGO run --target $TARGET           | grep "x = 42"
        env RUSTC_BOOTSTRAP=1 RUSTFLAGS="-C linker=arm-none-eabi-ld -C link-arg=-Tlink.x" \
            $BOOTSTRAP_CARGO run --target $TARGET --release | grep "x = 42"
    popd
popd
