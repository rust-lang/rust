#!/bin/sh
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

# Prep the SDK and emulator
#
# Note that the update process requires that we accept a bunch of licenses, and
# we can't just pipe `yes` into it for some reason, so we take the same strategy
# located in https://github.com/appunite/docker by just wrapping it in a script
# which apparently magically accepts the licenses.

mkdir sdk
curl https://dl.google.com/android/android-sdk_r24.4-linux.tgz | \
    tar xzf - -C sdk --strip-components=1

filter="platform-tools,android-18"
filter="$filter,sys-img-armeabi-v7a-android-18"

./accept-licenses.sh "android - update sdk -a --no-ui --filter $filter"

echo "no" | android create avd \
                --name arm-18 \
                --target android-18 \
                --abi armeabi-v7a
