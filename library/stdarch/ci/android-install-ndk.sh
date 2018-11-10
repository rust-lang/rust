#!/usr/bin/env sh
# Copyright 2016 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -ex

curl -O \
     https://dl.google.com/android/repository/android-ndk-r15b-linux-x86_64.zip
unzip -q android-ndk-r15b-linux-x86_64.zip

case "${1}" in
  aarch64)
    arch=arm64
    ;;

  i686)
    arch=x86
    ;;

  *)
    arch="${1}"
    ;;
esac;

android-ndk-r15b/build/tools/make_standalone_toolchain.py \
        --unified-headers \
        --install-dir "/android/ndk-${1}" \
        --arch "${arch}" \
        --api 24

rm -rf ./android-ndk-r15b-linux-x86_64.zip ./android-ndk-r15b
