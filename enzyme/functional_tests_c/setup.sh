#!/bin/bash

# NOTE(TFK): Uncomment for local testing.
export CLANG_BIN_PATH=./../../build-dbg/bin
export ENZYME_PLUGIN=./../mkdebug/Enzyme/LLVMEnzyme-7.so

mkdir -p build
$@
