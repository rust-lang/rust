#!/usr/bin/env sh

set -ex

export CARGO_NET_RETRY=5
export CARGO_NET_TIMEOUT=10

if rustup component add miri ; then
    cargo miri setup
    cargo miri test -- -- -Zunstable-options --exclude-should-panic
fi
