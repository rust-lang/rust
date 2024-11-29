#!/usr/bin/env sh

set -ex

: "${TARGET?The TARGET environment variable must be set.}"

case "${TARGET}" in
    x86_64-unknown-linux-gnu|x86_64-apple-darwin|aarch64-apple-darwin|x86_64-pc-windows-msvc)
        CROSS=0
        NO_STD=0
        ;;
    thumbv6m-none-eabi|loongarch64-unknown-linux-gnu)
        CROSS=1
        NO_STD=1
        ;;
    *)
        CROSS=1
        NO_STD=0
        ;;
esac

CARGO=cargo
if [ "${CROSS}" = "1" ]; then
    export CARGO_NET_RETRY=5
    export CARGO_NET_TIMEOUT=10

    cargo install --locked cross
    CARGO=cross
fi

if [ "${NO_STD}" = "1" ]; then
    # Unfortunately serde currently doesn't work without std due to a cargo bug.
    FEATURES="rustc-internal-api"
    OP="build"
else
    FEATURES="rustc-internal-api,serde,rayon"
    OP="test"
fi

if [ "${CHANNEL}" = "nightly" ]; then
    FEATURES="${FEATURES},nightly"
    export RUSTFLAGS="$RUSTFLAGS -D warnings"
fi

# Make sure we can compile without the default hasher
"${CARGO}" -vv build --target="${TARGET}" --no-default-features
"${CARGO}" -vv build --target="${TARGET}" --release --no-default-features

"${CARGO}" -vv ${OP} --target="${TARGET}"
"${CARGO}" -vv ${OP} --target="${TARGET}" --features "${FEATURES}"

"${CARGO}" -vv ${OP} --target="${TARGET}" --release
"${CARGO}" -vv ${OP} --target="${TARGET}" --release --features "${FEATURES}"

if [ "${CHANNEL}" = "nightly" ] && [ "${NO_STD}" != 1 ]; then
    # Run benchmark on native targets, build them on non-native ones:
    NO_RUN=""
    if [ "${CROSS}" = "1" ]; then
        NO_RUN="--no-run"
    fi

    "${CARGO}" -vv bench "${NO_RUN}" --features "${FEATURES}"
fi
