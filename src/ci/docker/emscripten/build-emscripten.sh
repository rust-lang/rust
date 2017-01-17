#!/bin/bash
# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -ex

curl https://s3.amazonaws.com/mozilla-games/emscripten/releases/emsdk-portable.tar.gz | \
      tar xzf -
source emsdk_portable/emsdk_env.sh
emsdk update
emsdk install --build=Release sdk-tag-1.37.1-32bit
emsdk activate --build=Release sdk-tag-1.37.1-32bit
