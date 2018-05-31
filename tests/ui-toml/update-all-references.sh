#!/bin/bash
#
# Copyright 2015 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# A script to update the references for all tests. The idea is that
# you do a run, which will generate files in the build directory
# containing the (normalized) actual output of the compiler. You then
# run this script, which will copy those files over. If you find
# yourself manually editing a foo.stderr file, you're doing it wrong.
#
# See all `update-references.sh`, if you just want to update a single test.

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "usage: $0"
fi

BUILD_DIR=$PWD/target/debug/test_build_base
MY_DIR=$(dirname $0)
cd $MY_DIR
find . -name '*.rs' | xargs ./update-references.sh $BUILD_DIR
