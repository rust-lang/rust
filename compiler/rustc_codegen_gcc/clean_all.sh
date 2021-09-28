#!/bin/bash --verbose
set -e

rm -rf target/ build_sysroot/{sysroot/,sysroot_src/,target/,Cargo.lock} perf.data{,.old}
rm -rf regex/ simple-raytracer/
