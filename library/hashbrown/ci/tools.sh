#!/usr/bin/env bash

set -ex

retry() {
    result=0
    count=1
    max=5
    while [ "$count" -le 3 ]; do
        [ "$result" -ne 0 ] && {
            printf "\nRetrying, %d of %d\n" $count $max >&2
        }
        "$@"
        result=$?
        [ $result -eq 0 ] && break
        count=$((count + 1))
        sleep 1
    done

    [ "$count" -gt 3 ] && {
        printf "\nFailed %d times.\n" $max >&2
    }

    return $result
}


if rustc --version | grep --quiet nightly ; then
    export RUSTDOCFLAGS="-Zunstable-options --check"
fi

cargo doc --no-deps --features serde,rayon

if retry rustup component add rustfmt ; then
    cargo fmt --all -- --check
fi

if retry rustup component add clippy ; then
    # we want all targets that have dedicated SIMD implementations,
    # plus one that uses the generic implementation,
    # since we can't lint cfg-disabled code
    TARGETS=()

    # associative array of cfgs from rustc
    declare -A CFG
    for cfg in $(rustc --print cfg); do
        # bash is very upset about quotes and equal signs in keys
        cfg="${cfg//\"/}"
        cfg="${cfg//=/_}"
        printf "\n%s\n" "$cfg" >&2
        CFG["$cfg"]=1
    done

    HOST_IS_GENERIC=1

    # SSE2
    if (( CFG[target_feature_sse2] && (CFG[target_arch_x86] || CFG[target_arch_x86_64]) )); then
        printf "\nHost target supports SSE2\n" >&2
        TARGETS+=(--target "$(rustc --print host-tuple)")
        HOST_IS_GENERIC=0
    elif retry rustup target add x86_64-unknown-linux-gnu; then
        printf "\nTesting x86_64-unknown-linux-gnu\n" >&2
        TARGETS+=(--target x86_64-unknown-linux-gnu)
    fi

    # NEON
    if (( CFG[target_arch_aarch64] && CFG[target_feature_neon] && CFG[target_endian_little] )); then
        printf "\nHost target supports NEON\n" >&2
        TARGETS+=(--target "$(rustc --print host-tuple)")
        HOST_IS_GENERIC=0
    elif retry rustup target add aarch64-unknown-linux-gnu; then
        printf "\nTesting aarch64-unknown-linux-gnu\n" >&2
        TARGETS+=(--target aarch64-unknown-linux-gnu)
    fi

    # LSX
    if (( CFG[target_arch_loongarch64] && CFG[target_feature_lsx] )); then
        printf "\nHost target supports LSX\n" >&2
        TARGETS+=(--target "$(rustc --print host-tuple)")
        HOST_IS_GENERIC=0
    elif retry rustup target add loongarch64-unknown-linux-gnu; then
        printf "\nTesting loongarch64-unknown-linux-gnu\n" >&2
        TARGETS+=(--target loongarch64-unknown-linux-gnu)
    fi

    # Generic
    if (( HOST_IS_GENERIC )); then
        printf "\nHost target support is generic\n" >&2
        TARGETS+=(--target "$(rustc --print host-tuple)")
    elif retry rustup target add i586-unknown-linux-gnu; then
        printf "\nTesting i586-unknown-linux-gnu\n" >&2
        TARGETS+=(--target i586-unknown-linux-gnu)
    fi

    cargo clippy --all --tests --features serde,rayon "${TARGETS[@]}" -- -D warnings
fi

if command -v taplo ; then
    taplo fmt --check --diff
fi

if command -v shellcheck ; then
    shellcheck --version
    shellcheck ci/*.sh
fi
