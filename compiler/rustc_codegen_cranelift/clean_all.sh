#!/bin/sh
set -e

rm -rf target/ build_system/target download/ build/ dist/

# Kept for now in case someone updates their checkout of cg_clif before running clean_all.sh
# FIXME remove at some point in the future
rm y.bin y.bin.dSYM y.exe y.pdb 2>/dev/null || true
rm -rf rand/ regex/ simple-raytracer/ portable-simd/ abi-checker/ abi-cafe/
rm -rf build_sysroot/{sysroot_src/,target/,compiler-builtins/,rustc_version}
