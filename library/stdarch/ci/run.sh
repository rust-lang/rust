#!/bin/sh

set -ex

: ${TARGET?"The TARGET environment variable must be set."}

# Tests are all super fast anyway, and they fault often enough on travis that
# having only one thread increases debuggability to be worth it.
export RUST_TEST_THREADS=1
export RUST_BACKTRACE=full
#export RUST_TEST_NOCAPTURE=1

RUSTFLAGS="$RUSTFLAGS --cfg stdsimd_strict"

# FIXME: on armv7 neon intrinsics require the neon target-feature to be
# unconditionally enabled.
# FIXME: powerpc (32-bit) must be compiled with altivec
# FIXME: on powerpc (32-bit) and powerpc64 (big endian) disable
# the instr tests.
case ${TARGET} in
    armv7*)
        export RUSTFLAGS="${RUSTFLAGS} -C target-feature=+neon"
        ;;
    powerpc-*)
        export RUSTFLAGS="${RUSTFLAGS} -C target-feature=+altivec"
        export STDSIMD_DISABLE_ASSERT_INSTR=1
        ;;
    powerpc64-*)
        export STDSIMD_DISABLE_ASSERT_INSTR=1
        ;;
    *)
        ;;
esac

echo "RUSTFLAGS=${RUSTFLAGS}"
echo "FEATURES=${FEATURES}"
echo "OBJDUMP=${OBJDUMP}"
echo "STDSIMD_DISABLE_ASSERT_INSTR=${STDSIMD_DISABLE_ASSERT_INSTR}"

cargo_test() {
    cmd="cargo test --target=$TARGET $1"
    cmd="$cmd -p coresimd -p stdsimd"
    cmd="$cmd -- $2"
    $cmd
}

cargo_test
cargo_test "--release"

# Test x86 targets compiled with AVX.
case ${TARGET} in
    x86*)
        RUSTFLAGS="${RUSTFLAGS} -C target-feature=+avx"
        export STDSIMD_DISABLE_ASSERT_INSTR=1
        cargo_test "--release"
        ;;
    *)
        ;;
esac
