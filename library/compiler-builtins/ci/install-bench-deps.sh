#!/bin/sh
# Install needed dependencies for gungraun.

sudo apt-get update
sudo apt-get install -y valgrind gdb libc6-dbg # Needed for gungraun
rustup update "$BENCHMARK_RUSTC" --no-self-update
rustup default "$BENCHMARK_RUSTC"
# Install the version of gungraun-runner that is specified in Cargo.toml
gungraun_version="$(cargo metadata --format-version=1 --features icount |
    jq -r '.packages[] | select(.name == "gungraun").version')"
cargo binstall -y gungraun-runner --version "$gungraun_version"
