#!/bin/bash
set -ex

# We need Tree Borrows as some of our raw pointer patterns are not
# compatible with Stacked Borrows.
export MIRIFLAGS="-Zmiri-tree-borrows"

# One target that sets `mem-unaligned` and one that does not,
# and a big-endian target.
TARGETS=(x86_64-unknown-linux-gnu
    armv7-unknown-linux-gnueabihf
    s390x-unknown-linux-gnu)
for TARGET in "${TARGETS[@]}"; do
    # Only run the `mem` tests to avoid this taking too long.
    cargo miri test --manifest-path builtins-test/Cargo.toml --features no-asm --target $TARGET -- mem
done
