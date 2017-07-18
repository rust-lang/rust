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

URL=https://dl.google.com/android/repository

download_ndk() {
    mkdir -p /android/ndk
    cd /android/ndk
    curl -sO $URL/$1
    unzip -q $1
    rm $1
    mv android-ndk-* ndk
}

make_standalone_toolchain() {
    # See https://developer.android.com/ndk/guides/standalone_toolchain.htm
    python2.7 /android/ndk/ndk/build/tools/make_standalone_toolchain.py \
        --install-dir /android/ndk/$1-$2 \
        --arch $1 \
        --api $2
}

remove_ndk() {
    rm -rf /android/ndk/ndk
}

download_and_make_toolchain() {
    download_ndk $1 && \
    make_standalone_toolchain $2 $3 && \
    remove_ndk
}
