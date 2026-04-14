#!/bin/bash
set -e

echo "Running masked compositing benchmark smoke test..."

# Just verify that bench_blit builds for x86_64
# This ensures the benchmark code compiles and links correctly
echo "Building bench_blit for x86_64..."
cargo build -Z build-std=core,alloc -Z build-std-features=compiler-builtins-mem \
    --target targets/x86_64-unknown-thingos.json \
    -p bench_blit

echo "Benchmark smoke test PASSED."
