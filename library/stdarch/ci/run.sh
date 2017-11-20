#!/bin/sh

set -ex

# Tests are all super fast anyway, and they fault often enough on travis that
# having only one thread increases debuggability to be worth it.
export RUST_TEST_THREADS=1

# FIXME(rust-lang-nursery/stdsimd#120) run-time feature detection for ARM Neon
case ${TARGET} in
    aarch*)
        export RUSTFLAGS="${RUSTFLAGS} -C target-feature=+neon"
        ;;
    *)
        ;;
esac

FEATURES="strict,$FEATURES"

echo "RUSTFLAGS=${RUSTFLAGS}"
echo "FEATURES=${FEATURES}"
echo "OBJDUMP=${OBJDUMP}"

cargo_test() {
    cmd="cargo test --all --target=$TARGET --features $FEATURES --verbose $1 -- --nocapture $2"
    $cmd
}

cargo_test
cargo_test "--release"
