#!/bin/bash

set -eux

# Run Cargo tests using the stable Rust compiler

TOOLCHAIN=stable

rustup install --no-self-update "${TOOLCHAIN}"

# We run Cargo tests directly, because we need to override the compiler used by Cargo's tests
# to stable. That would be non-trivial with the way bootstrap is currently set up.
cd src/tools/cargo || exit 1
cargo +${TOOLCHAIN} test --workspace --exclude benchsuite --exclude resolver-tests
