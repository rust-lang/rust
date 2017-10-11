#!/bin/sh

set -ex

# FIXME(rust-lang/rust#45201) shouldn't need to specify one codegen unit
export RUSTFLAGS="$RUSTFLAGS -C codegen-units=1"

cargo test --target $TARGET
cargo test --release --target $TARGET
