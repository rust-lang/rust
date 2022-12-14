#!/usr/bin/env bash
set -e

rm -rf build_sysroot/{sysroot_src/,target/,compiler-builtins/,rustc_version}
rm -rf target/ build/ perf.data{,.old} y.bin
rm -rf download/

# Kept for now in case someone updates their checkout of cg_clif before running clean_all.sh
# FIXME remove at some point in the future
rm -rf rand/ regex/ simple-raytracer/ portable-simd/ abi-checker/ abi-cafe/
