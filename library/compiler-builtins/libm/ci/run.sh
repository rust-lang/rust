#!/bin/sh

set -ex
TARGET=$1

cargo test --target $TARGET
cargo test --target $TARGET --release

# FIXME(#4) overflow checks in non-release currently cause issues
#cargo test --features 'checked musl-reference-tests' --target $TARGET

cargo test --features 'checked musl-reference-tests' --target $TARGET --release
