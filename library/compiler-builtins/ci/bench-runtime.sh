#!/bin/sh
# Run wall time benchmarks as we do on CI.

# Always use the same seed for benchmarks. Ideally we should switch to a
# non-random generator.
export LIBM_SEED=benchesbenchesbenchesbencheswoo!
cargo bench --package libm-test \
    --no-default-features \
    --features short-benchmarks,build-musl,libm/force-soft-floats
