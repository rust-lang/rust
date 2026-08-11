#!/bin/bash

set -euo pipefail

BUILD_DIR=$(realpath ./build/x86_64-unknown-linux-gnu)

# Install the latest version of cargo-semver-checks, so that once the JSON doc format changes,
# we will eventually get a csc version that supports it
# Speed up compilation by reducing optimizations settings a bit
RUSTC="${BUILD_DIR}"/stage0/bin/rustc \
CARGO_PROFILE_RELEASE_LTO=false \
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
"${BUILD_DIR}"/stage0/bin/cargo install cargo-semver-checks --locked

# Provide path to cargo-semver-checks
export PATH=${PATH}:/cargo/bin

# Explicitly compute the baseline commit (the first git parent, which is the latest upstream main
# commit), so that it is shown in the commit log and so that the command can be easily reproduced
# locally.
PARENT=$(git rev-parse HEAD^1)

# Run the test
python3 ../x.py test std-semver-check --set rust.stdlib-semver-baseline=${PARENT}
