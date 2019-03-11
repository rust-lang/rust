#!/usr/bin/env sh

set -ex

: "${TARGET?The TARGET environment variable must be set.}"

FEATURES="rayon,serde"
if [ "${TRAVIS_RUST_VERSION}" = "nightly" ]; then
    FEATURES="${FEATURES},nightly"
    export RUSTFLAGS="$RUSTFLAGS -D warnings"
fi

CARGO=cargo
if [ "${CROSS}" = "1" ]; then
    export CARGO_NET_RETRY=5
    export CARGO_NET_TIMEOUT=10

    cargo install cross
    CARGO=cross
fi

export RUSTFLAGS="$RUSTFLAGS --cfg hashbrown_deny_warnings"

"${CARGO}" -vv test --target="${TARGET}"
"${CARGO}" -vv test --target="${TARGET}" --features "${FEATURES}"

"${CARGO}" -vv test --target="${TARGET}" --release
"${CARGO}" -vv test --target="${TARGET}" --release --features "${FEATURES}"

if [ "${TRAVIS_RUST_VERSION}" = "nightly" ]; then
    # Run benchmark on native targets, build them on non-native ones:
    NO_RUN=""
    if [ "${CROSS}" = "1" ]; then
        NO_RUN="--no-run"
    fi

    "${CARGO}" -vv bench "${NO_RUN}" --features "${FEATURES}"
fi
