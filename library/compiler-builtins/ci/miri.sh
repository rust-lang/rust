#!/bin/bash
set -eux

# We need Tree Borrows as some of our raw pointer patterns are not
# compatible with Stacked Borrows.
export MIRIFLAGS="-Zmiri-tree-borrows"

# One target that sets `mem_unaligned` and one that does not,
# and a big-endian target.
targets=(
    x86_64-unknown-linux-gnu
    armv7-unknown-linux-gnueabihf
    s390x-unknown-linux-gnu
)
for target in "${targets[@]}"; do
    # Only run the `mem` tests to avoid this taking too long. Disable default
    # features to turn off `arch` and avoid inline assembly.
    cargo miri test \
        --manifest-path builtins-test/Cargo.toml \
        --no-default-features \
        --target "$target" \
        -- mem
done
