#!/usr/bin/env sh

set -ex

export CARGO_NET_RETRY=5
export CARGO_NET_TIMEOUT=10

rustup toolchain install nightly --component miri
rustup override set nightly
cargo miri setup

cargo miri test
