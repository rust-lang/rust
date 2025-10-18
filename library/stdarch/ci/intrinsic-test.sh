#!/usr/bin/env sh

set -ex

: "${TARGET?The TARGET environment variable must be set.}"

# Tests are all super fast anyway, and they fault often enough on travis that
# having only one thread increases debuggability to be worth it.
#export RUST_BACKTRACE=full
#export RUST_TEST_NOCAPTURE=1
#export RUST_TEST_THREADS=1

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
echo "STDARCH_DISABLE_ASSERT_INSTR=${STDARCH_DISABLE_ASSERT_INSTR}"
echo "STDARCH_TEST_EVERYTHING=${STDARCH_TEST_EVERYTHING}"
echo "STDARCH_TEST_SKIP_FEATURE=${STDARCH_TEST_SKIP_FEATURE}"
echo "STDARCH_TEST_SKIP_FUNCTION=${STDARCH_TEST_SKIP_FUNCTION}"
echo "PROFILE=${PROFILE}"

INTRINSIC_TEST="--manifest-path=crates/intrinsic-test/Cargo.toml"

# Test targets compiled with extra features.
case ${TARGET} in

    x86_64* | i686*)
        export STDARCH_DISABLE_ASSERT_INSTR=1
        ;;

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
     *)
        ;;
esac
