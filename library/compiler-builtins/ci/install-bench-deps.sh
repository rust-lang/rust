#!/bin/bash
# Install needed dependencies for gungraun.

set -eux

target="${1:-}"

# Needed for gungraun
deps=(valgrind gdb libc6-dbg)

[[ "$target" = *"i686"* ]] && deps+=(gcc-multilib)

sudo apt-get update
sudo apt-get install -y "${deps[@]}"

rustup update "$BENCHMARK_RUSTC" --no-self-update
rustup default "$BENCHMARK_RUSTC"
[ -n "$target" ] && rustup target add "$target"

# Install the version of gungraun-runner that is specified in Cargo.toml
gungraun_version="$(cargo metadata --format-version=1 --features icount |
    jq -r '.packages[] | select(.name == "gungraun").version')"
cargo binstall -y gungraun-runner --version "$gungraun_version"
