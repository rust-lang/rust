#!/usr/bin/env sh

set -ex

cargo build
cargo test --verbose

# install
mkdir -p ~/rust/cargo/bin
cp target/debug/cargo-semver ~/rust/cargo/bin
cp target/debug/rust-semverver ~/rust/cargo/bin

# become semververver
current_version=$(grep -e '^version = .*$' Cargo.toml | cut -d ' ' -f 3)
PATH=~/rust/cargo/bin:$PATH cargo semver | tee semver_out
(head -n 1 semver_out | grep "\\-> $current_version") || (echo "versioning mismatch" && return 1)
