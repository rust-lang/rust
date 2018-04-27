#!/bin/sh

set -ex

: ${TARGET?"The TARGET environment variable must be set."}

# Tests are all super fast anyway, and they fault often enough on travis that
# having only one thread increases debuggability to be worth it.
export RUST_TEST_THREADS=1
#export RUST_BACKTRACE=full
#export RUST_TEST_NOCAPTURE=1

RUSTFLAGS="$RUSTFLAGS --cfg stdsimd_strict"

# FIXME: on armv7 neon intrinsics require the neon target-feature to be
# unconditionally enabled.
case ${TARGET} in
    armv7*)
        export RUSTFLAGS="${RUSTFLAGS} -C target-feature=+neon"
        ;;
    *)
        ;;
esac

echo "RUSTFLAGS=${RUSTFLAGS}"
echo "FEATURES=${FEATURES}"
echo "OBJDUMP=${OBJDUMP}"

cargo_test() {
    cmd="cargo test --target=$TARGET $1"
    cmd="$cmd -p coresimd -p stdsimd"
    cmd="$cmd -- $2"
    $cmd
}

cargo_test
cargo_test "--release"
