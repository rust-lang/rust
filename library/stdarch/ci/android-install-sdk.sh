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

# Prep the SDK and emulator
#
# Note that the update process requires that we accept a bunch of licenses, and
# we can't just pipe `yes` into it for some reason, so we take the same strategy
# located in https://github.com/appunite/docker by just wrapping it in a script
# which apparently magically accepts the licenses.

mkdir sdk
curl --retry 5 https://dl.google.com/android/repository/sdk-tools-linux-3859397.zip -O
unzip -d sdk sdk-tools-linux-3859397.zip

case "$1" in
  arm | armv7)
    abi=armeabi-v7a
    ;;

  aarch64)
    abi=arm64-v8a
    ;;

  i686)
    abi=x86
    ;;

  x86_64)
    abi=x86_64
    ;;

  *)
    echo "invalid arch: $1"
    exit 1
    ;;
esac;

# --no_https avoids
# javax.net.ssl.SSLHandshakeException: sun.security.validator.ValidatorException: No trusted certificate found
echo "yes" | \
    ./sdk/tools/bin/sdkmanager --no_https \
        "emulator" \
        "platform-tools" \
        "platforms;android-24" \
        "system-images;android-24;default;$abi"

echo "no" |
    ./sdk/tools/bin/avdmanager create avd \
        --name "${1}" \
        --package "system-images;android-24;default;$abi"
