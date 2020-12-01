#!/bin/bash --verbose
set -e

rm -rf target/ build/ build_sysroot/{sysroot_src/,target/} perf.data{,.old}
rm -rf rand/ regex/ simple-raytracer/
