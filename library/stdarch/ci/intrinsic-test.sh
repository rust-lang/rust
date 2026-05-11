#!/usr/bin/env sh

set -ex

if [ $# -lt 2 ]; then
    >&2 echo "Usage: $0 <TARGET> <CC>"
    exit 1
fi

case ${2} in
    clang)
        export CC="${CLANG_PATH}"
        ;;
    gcc)
        export CC="${GCC_PATH}"
        ;;
    *)
        >&2 echo "Unknown compiler: ${2}"
        exit 1
        ;;
esac

export RUSTFLAGS="${RUSTFLAGS} -D warnings -Z merge-functions=disabled -Z verify-llvm-ir"
export PROFILE="${PROFILE:="release"}"

echo "RUSTFLAGS=${RUSTFLAGS}"
echo "PROFILE=${PROFILE}"

INTRINSIC_TEST="--manifest-path=crates/intrinsic-test/Cargo.toml"

case ${1} in
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

case "${1}" in
    x86_64-unknown-linux-gnu*)
        env -u CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER \
            cargo run "${INTRINSIC_TEST}" --release \
            --bin intrinsic-test -- intrinsics_data/x86-intel.xml \
            --skip "crates/intrinsic-test/missing_${ARCH}_common.txt" \
            --skip "crates/intrinsic-test/missing_${ARCH}_${2}.txt" \
            --target "${1}" \
            --cc-arg-style "${2}"
        ;;
    *)
        cargo run "${INTRINSIC_TEST}" --release \
            --bin intrinsic-test -- intrinsics_data/arm_intrinsics.json \
            --skip "crates/intrinsic-test/missing_${ARCH}_common.txt" \
            --skip "crates/intrinsic-test/missing_${ARCH}_${2}.txt" \
            --target "${1}" \
            --cc-arg-style "${2}"
        ;;
esac

cargo test --manifest-path=rust_programs/Cargo.toml --target "${1}" --profile "${PROFILE}" --tests
