#!/usr/bin/env sh

if [ $# -lt 2 ]; then
    >&2 echo "Usage: $0 <TARGET> <CC> <..args for \`cargo test\`..>"
    exit 1
fi

set -ex

# Pop both arguments and leave "$@" as containing args to be forwarded to `cargo test`
TARGET="$1"
shift
CC_KIND="$1"
shift

case ${CC_KIND} in
    clang)
        export CC="${CLANG_PATH}"
        CC_ARG_STYLE=clang
        ;;
    gcc)
        export CC="${GCC_PATH}"
        CC_ARG_STYLE=gcc
        ;;
    icx)
        export CC="${ICX_PATH}"
        # `icx` uses clang-style arguments
        CC_ARG_STYLE=clang
        ;;
    *)
        >&2 echo "Unknown compiler: ${CC_KIND}"
        exit 1
        ;;
esac

export RUSTFLAGS="${RUSTFLAGS} -D warnings -Z merge-functions=disabled -Z verify-llvm-ir"
export PROFILE="${PROFILE:="release"}"

echo "RUSTFLAGS=${RUSTFLAGS}"
echo "PROFILE=${PROFILE}"

INTRINSIC_TEST="--manifest-path=crates/intrinsic-test/Cargo.toml"

case ${TARGET} in
    aarch64_be*)
        export CFLAGS="-I${AARCH64_BE_TOOLCHAIN}/aarch64_be-none-linux-gnu/libc/usr/include --sysroot={AARCH64_BE_TOOLCHAIN}/aarch64_be-none-linux-gnu/libc -Wno-nonportable-vector-initialization"
        ARCH=aarch64_be
        ;;

    aarch64*)
        export CFLAGS="-I/usr/aarch64-linux-gnu/include/"
        ARCH=aarch64
        ;;

    armv7*)
        export CFLAGS="-I/usr/arm-linux-gnueabihf/include/"
        ARCH=arm
        ;;

    x86_64*)
        export CFLAGS="-I/usr/include/x86_64-linux-gnu/"
        ARCH=x86
        ;;
    *)
        ;;

esac

case "${TARGET}" in
    x86_64-unknown-linux-gnu*)
        env -u CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER \
            cargo run "${INTRINSIC_TEST}" --release \
            --bin intrinsic-test -- intrinsics_data/x86-intel.xml \
            --skip "crates/intrinsic-test/missing_${ARCH}_common.txt" \
            --skip "crates/intrinsic-test/missing_${ARCH}_${CC_KIND}.txt" \
            --target "${TARGET}" \
            --cc-arg-style "${CC_ARG_STYLE}"
        ;;
    *)
        cargo run "${INTRINSIC_TEST}" --release \
            --bin intrinsic-test -- intrinsics_data/arm_intrinsics.json \
            --skip "crates/intrinsic-test/missing_${ARCH}_common.txt" \
            --skip "crates/intrinsic-test/missing_${ARCH}_${CC_KIND}.txt" \
            --target "${TARGET}" \
            --cc-arg-style "${CC_ARG_STYLE}"
        ;;
esac

cargo test --manifest-path=rust_programs/Cargo.toml --target "${TARGET}" --profile "${PROFILE}" \
    --tests "$@"
