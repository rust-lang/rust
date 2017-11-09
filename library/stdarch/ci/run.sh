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
FEATURES_STD="${FEATURES},std"

echo "RUSTFLAGS=${RUSTFLAGS}"
echo "FEATURES=${FEATURES}"

cargo test --target $TARGET --features $FEATURES
cargo test --release --target $TARGET --features $FEATURES

cargo test --target $TARGET --features $FEATURES_STD
cargo test --release --target $TARGET --features $FEATURES_STD
