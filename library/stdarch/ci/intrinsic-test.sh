#!/usr/bin/env sh

set -ex

: "${TARGET?The TARGET environment variable must be set.}"

export RUSTFLAGS="${RUSTFLAGS} -D warnings -Z merge-functions=disabled -Z verify-llvm-ir"
export HOST_RUSTFLAGS="${RUSTFLAGS}"
export PROFILE="${PROFILE:="--profile=release"}"

case ${TARGET} in
    # On 32-bit use a static relocation model which avoids some extra
    # instructions when dealing with static data, notably allowing some
    # instruction assertion checks to pass below the 20 instruction limit. If
    # this is the default, dynamic, then too many instructions are generated
    # when we assert the instruction for a function and it causes tests to fail.
    i686-* | i586-*)
        export RUSTFLAGS="${RUSTFLAGS} -C relocation-model=static"
        ;;
    # Some x86_64 targets enable by default more features beyond SSE2,
    # which cause some instruction assertion checks to fail.
    x86_64-*)
        export RUSTFLAGS="${RUSTFLAGS} -C target-feature=-sse3"
        ;;
    #Unoptimized build uses fast-isel which breaks with msa
    mips-* | mipsel-*)
	export RUSTFLAGS="${RUSTFLAGS} -C llvm-args=-fast-isel=false"
	;;
    armv7-*eabihf | thumbv7-*eabihf)
        export RUSTFLAGS="${RUSTFLAGS} -Ctarget-feature=+neon"
        ;;
    # Some of our test dependencies use the deprecated `gcc` crates which
    # doesn't detect RISC-V compilers automatically, so do it manually here.
    riscv*)
        export RUSTFLAGS="${RUSTFLAGS} -Ctarget-feature=+zk,+zks,+zbb,+zbc"
        ;;
esac

echo "RUSTFLAGS=${RUSTFLAGS}"
echo "OBJDUMP=${OBJDUMP}"
echo "PROFILE=${PROFILE}"

INTRINSIC_TEST="--manifest-path=crates/intrinsic-test/Cargo.toml"

# Test targets compiled with extra features.
case ${TARGET} in
    # Setup aarch64 & armv7 specific variables, the runner, along with some
    # tests to skip
    aarch64-unknown-linux-gnu*)
        TEST_CPPFLAGS="-fuse-ld=lld -I/usr/aarch64-linux-gnu/include/ -I/usr/aarch64-linux-gnu/include/c++/9/aarch64-linux-gnu/"
        TEST_SKIP_INTRINSICS=crates/intrinsic-test/missing_aarch64.txt
        TEST_CXX_COMPILER="clang++"
        TEST_RUNNER="${CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER}"
        ;;

    aarch64_be-unknown-linux-gnu*)
        TEST_CPPFLAGS="-fuse-ld=lld"
        TEST_SKIP_INTRINSICS=crates/intrinsic-test/missing_aarch64.txt
        TEST_CXX_COMPILER="clang++"
        TEST_RUNNER="${CARGO_TARGET_AARCH64_BE_UNKNOWN_LINUX_GNU_RUNNER}"
        ;;

    armv7-unknown-linux-gnueabihf*)
        TEST_CPPFLAGS="-fuse-ld=lld -I/usr/arm-linux-gnueabihf/include/ -I/usr/arm-linux-gnueabihf/include/c++/9/arm-linux-gnueabihf/"
        TEST_SKIP_INTRINSICS=crates/intrinsic-test/missing_arm.txt
        TEST_CXX_COMPILER="clang++"
        TEST_RUNNER="${CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_RUNNER}"
        ;;

    x86_64-unknown-linux-gnu*)
        TEST_CPPFLAGS="-fuse-ld=lld -I/usr/include/x86_64-linux-gnu/"
        TEST_CXX_COMPILER="clang++"
        TEST_RUNNER="${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER}"
        TEST_SKIP_INTRINSICS=crates/intrinsic-test/missing_x86.txt
        TEST_SAMPLE_INTRINSICS_PERCENTAGE=5
        ;;
    *)
        ;;

esac

# Arm specific
case "${TARGET}" in
    aarch64-unknown-linux-gnu*|armv7-unknown-linux-gnueabihf*)
        CPPFLAGS="${TEST_CPPFLAGS}" RUSTFLAGS="${HOST_RUSTFLAGS}" RUST_LOG=warn \
            cargo run "${INTRINSIC_TEST}" "${PROFILE}" \
            --bin intrinsic-test -- intrinsics_data/arm_intrinsics.json \
            --runner "${TEST_RUNNER}" \
            --cppcompiler "${TEST_CXX_COMPILER}" \
            --skip "${TEST_SKIP_INTRINSICS}" \
            --target "${TARGET}"
        ;;

    aarch64_be-unknown-linux-gnu*)
        CPPFLAGS="${TEST_CPPFLAGS}" RUSTFLAGS="${HOST_RUSTFLAGS}" RUST_LOG=warn \
            cargo run "${INTRINSIC_TEST}" "${PROFILE}"  \
            --bin intrinsic-test -- intrinsics_data/arm_intrinsics.json \
            --runner "${TEST_RUNNER}" \
            --cppcompiler "${TEST_CXX_COMPILER}" \
            --skip "${TEST_SKIP_INTRINSICS}" \
            --target "${TARGET}" \
            --linker "${CARGO_TARGET_AARCH64_BE_UNKNOWN_LINUX_GNU_LINKER}" \
            --cxx-toolchain-dir "${AARCH64_BE_TOOLCHAIN}"
        ;;

    x86_64-unknown-linux-gnu*)
        # `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER` is not necessary for `intrinsic-test`
        # because the binary needs to run directly on the host.
        # Hence the use of `env -u`.
        env -u CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER \
            CPPFLAGS="${TEST_CPPFLAGS}" RUSTFLAGS="${HOST_RUSTFLAGS}" \
            RUST_LOG=warn RUST_BACKTRACE=1 \
            cargo run "${INTRINSIC_TEST}" "${PROFILE}"  \
            --bin intrinsic-test -- intrinsics_data/x86-intel.xml \
            --runner "${TEST_RUNNER}" \
            --skip "${TEST_SKIP_INTRINSICS}" \
            --cppcompiler "${TEST_CXX_COMPILER}" \
            --target "${TARGET}" \
            --sample-percentage "${TEST_SAMPLE_INTRINSICS_PERCENTAGE}"
        ;;
     *)
        ;;
esac
