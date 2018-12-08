#!/usr/bin/env sh

set -ex

export RUST_TEST_THREADS=1
export RUST_BACKTRACE=full
#export RUST_TEST_NOCAPTURE=1

cargo build
cargo test --verbose -- --nocapture

# install
mkdir -p ~/rust/cargo/bin
cp target/debug/cargo-semver ~/rust/cargo/bin
cp target/debug/rust-semverver ~/rust/cargo/bin

# become semververver
PATH=~/rust/cargo/bin:$PATH cargo semver | tee semver_out
current_version="$(grep -e '^version = .*$' Cargo.toml | cut -d ' ' -f 3)"
current_version="${current_version%\"}"
current_version="${current_version#\"}"
result="$(head -n 1 semver_out)"
if echo "$result" | grep -- "-> $current_version"; then
    echo "version ok"
    exit 0
else
    echo "versioning mismatch"
    cat semver_out
    echo "versioning mismatch"
    exit 1
fi
