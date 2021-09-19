#!/usr/bin/env bash
set -e

rm -rf build_sysroot/{sysroot_src/,target/,compiler-builtins/,rustc_version}
rm -rf target/ build/ perf.data{,.old}
rm -rf rand/ regex/ simple-raytracer/ portable-simd/
