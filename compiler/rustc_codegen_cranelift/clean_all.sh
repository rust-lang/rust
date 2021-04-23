#!/usr/bin/env bash
set -e

rm -rf target/ build/ build_sysroot/{sysroot_src/,target/,compiler-builtins/} perf.data{,.old}
rm -rf rand/ regex/ simple-raytracer/
