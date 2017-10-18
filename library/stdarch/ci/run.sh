#!/bin/sh

set -ex

# FIXME(rust-lang/rust#45201) shouldn't need to specify one codegen unit
export RUSTFLAGS="$RUSTFLAGS -C codegen-units=1"

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

echo "RUSTFLAGS=${RUSTFLAGS}"

cargo test --target $TARGET
cargo test --release --target $TARGET
