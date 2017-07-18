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

download_sdk() {
    mkdir -p /android/sdk
    cd /android/sdk
    curl -sO $URL/$1
    unzip -q $1
    rm -rf $1
}

download_sysimage() {
    # See https://developer.android.com/studio/tools/help/android.html
    abi=$1
    api=$2

    filter="platform-tools,android-$api"
    filter="$filter,sys-img-$abi-android-$api"

    # Keep printing yes to accept the licenses
    while true; do echo yes; sleep 10; done | \
        /android/sdk/tools/android update sdk -a --no-ui \
            --filter "$filter"
}

create_avd() {
    # See https://developer.android.com/studio/tools/help/android.html
    abi=$1
    api=$2

    echo no | \
        /android/sdk/tools/android create avd \
            --name $abi-$api \
            --target android-$api \
            --abi $abi
}

download_and_create_avd() {
    download_sdk $1
    download_sysimage $2 $3
    create_avd $2 $3
}
