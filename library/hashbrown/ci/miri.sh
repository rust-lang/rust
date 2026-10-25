#!/usr/bin/env bash

set -ex

export CARGO_NET_RETRY=5
export CARGO_NET_TIMEOUT=10

rustup toolchain install nightly --component miri
rustup override set nightly
cargo miri setup

TARGETS=(
    # SSE impl
    --target x86_64-unknown-linux-gnu
    # NEON impl
    --target aarch64-unknown-linux-gnu
    # generic impl
    --target riscv64a23-unknown-linux-gnu
)
MIRIFLAGS="$MIRIFLAGS -Zmiri-strict-provenance" cargo miri test --features nightly "${TARGETS[@]}"
