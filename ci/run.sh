#!/usr/bin/env sh

set -ex

# Note: this is required for correctness,
# otherwise executing multiple "full" tests in parallel
# of the same library can alter results.
export RUST_TEST_THREADS=1
export RUST_BACKTRACE=full
#export RUST_TEST_NOCAPTURE=1

cargo build
cargo test --verbose -- --nocapture

case "${TRAVIS_OS_NAME}" in
    *"linux"*)
        TEST_TARGET=x86_64-unknown-linux-gnu cargo test --verbose -- --nocapture
        ;;
    *"windows"*)
        TEST_TARGET=x86_64-pc-windows-msvc cargo test --verbose -- --nocapture
        ;;
    *"macos"*)
        TEST_TARGET=x86_64-apple-darwin cargo test --verbose -- --nocapture
        ;;
esac

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
