#!/bin/sh

set -ex
TARGET=$1

cargo build --target $TARGET
cargo test --target $TARGET
cargo build --target $TARGET --release
cargo test --target $TARGET --release

cargo test --features 'checked musl-reference-tests' --target $TARGET
cargo test --features 'checked musl-reference-tests' --target $TARGET --release
