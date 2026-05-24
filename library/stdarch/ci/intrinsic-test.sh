#!/usr/bin/env sh

set -ex

: "${TARGET?The TARGET environment variable must be set.}"

export RUSTFLAGS="${RUSTFLAGS} -D warnings -Z merge-functions=disabled -Z verify-llvm-ir"
export PROFILE="${PROFILE:="release"}"

echo "RUSTFLAGS=${RUSTFLAGS}"
echo "PROFILE=${PROFILE}"

INTRINSIC_TEST="--manifest-path=crates/intrinsic-test/Cargo.toml"

export CC="clang"

case ${TARGET} in
    aarch64_be*)
        export CFLAGS="-I${AARCH64_BE_TOOLCHAIN}/aarch64_be-none-linux-gnu/libc/usr/include --sysroot={AARCH64_BE_TOOLCHAIN}/aarch64_be-none-linux-gnu/libc -Wno-nonportable-vector-initialization"
        TEST_SKIP_INTRINSICS=crates/intrinsic-test/missing_aarch64_be.txt
        ;;

    aarch64*)
        export CFLAGS="-I/usr/aarch64-linux-gnu/include/"
        TEST_SKIP_INTRINSICS=crates/intrinsic-test/missing_aarch64.txt
        ;;

    armv7*)
        export CFLAGS="-I/usr/arm-linux-gnueabihf/include/"
        TEST_SKIP_INTRINSICS=crates/intrinsic-test/missing_arm.txt
        ;;

    x86_64*)
        export CFLAGS="-I/usr/include/x86_64-linux-gnu/"
        TEST_SKIP_INTRINSICS=crates/intrinsic-test/missing_x86.txt
        ;;
    *)
        ;;

esac

case "${TARGET}" in
    x86_64-unknown-linux-gnu*)
        env -u CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER \
            cargo run "${INTRINSIC_TEST}" --release \
            --bin intrinsic-test -- intrinsics_data/x86-intel.xml \
            --skip "${TEST_SKIP_INTRINSICS}" \
            --target "${TARGET}"

        echo "${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER}"
        ;;
    *)
        cargo run "${INTRINSIC_TEST}" --release \
            --bin intrinsic-test -- intrinsics_data/arm_intrinsics.json \
            --skip "${TEST_SKIP_INTRINSICS}" \
            --target "${TARGET}"
        ;;
esac

cargo test --manifest-path=rust_programs/Cargo.toml --target "${TARGET}" --profile "${PROFILE}" --tests
