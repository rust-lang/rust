#!/usr/bin/env sh

set -ex

export CARGO_NET_RETRY=5
export CARGO_NET_TIMEOUT=10

if cargo install --force --git https://github.com/rust-lang/miri miri ; then
    cargo miri setup
    cargo miri test
fi
