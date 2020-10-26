#!/bin/bash --verbose
set -e

rm -rf target/ build_sysroot/{sysroot/,sysroot_src/,target/} perf.data{,.old}
rm -rf rand/ regex/ simple-raytracer/
