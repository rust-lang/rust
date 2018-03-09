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

URL=https://dl.google.com/android/repository/sys-img/android

main() {
    local arch=$1
    local name=$2
    local dest=/system
    local td=$(mktemp -d)

    apt-get install --no-install-recommends e2tools

    pushd $td
    curl -O $URL/$name
    unzip -q $name

    local system=$(find . -name system.img)
    mkdir -p $dest/{bin,lib,lib64}

    # Extract android linker and libraries to /system
    # This allows android executables to be run directly (or with qemu)
    if [ $arch = "x86_64" -o $arch = "arm64" ]; then
        e2cp -p $system:/bin/linker64 $dest/bin/
        e2cp -p $system:/lib64/libdl.so $dest/lib64/
        e2cp -p $system:/lib64/libc.so $dest/lib64/
        e2cp -p $system:/lib64/libm.so $dest/lib64/
    else
        e2cp -p $system:/bin/linker $dest/bin/
        e2cp -p $system:/lib/libdl.so $dest/lib/
        e2cp -p $system:/lib/libc.so $dest/lib/
        e2cp -p $system:/lib/libm.so $dest/lib/
    fi

    # clean up
    apt-get purge --auto-remove -y e2tools

    popd

    rm -rf $td
}

main "${@}"
