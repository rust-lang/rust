#!/bin/bash

# NOTE(TFK): Uncomment for local testing.
export CLANG_BIN_PATH=./../../llvm/build/bin
export ENZYME_PLUGIN=./../build/Enzyme/LLVMEnzyme-7.so

mkdir -p build
$@
