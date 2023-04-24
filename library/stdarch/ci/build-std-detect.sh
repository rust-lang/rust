#!/usr/bin/env bash

# Build std_detect on non-Linux & non-x86 targets.
#
# In std_detect, non-x86 targets have OS-specific implementations,
# but we can test only Linux in CI. This script builds targets supported
# by std_detect but cannot be tested in CI.

set -ex
cd "$(dirname "$0")"/..

targets=(
    # Android
    aarch64-linux-android
    arm-linux-androideabi

    # FreeBSD
    aarch64-unknown-freebsd
    armv6-unknown-freebsd
    powerpc-unknown-freebsd
    powerpc64-unknown-freebsd

    # OpenBSD
    aarch64-unknown-openbsd

    # Windows
    aarch64-pc-windows-msvc
)

rustup component add rust-src # for -Z build-std

cd crates/std_detect
for target in "${targets[@]}"; do
    if rustup target add "${target}" &>/dev/null; then
        cargo build --target "${target}"
    else
        # tier 3 targets requires -Z build-std.
        cargo build -Z build-std="core,alloc" --target "${target}"
    fi
done
