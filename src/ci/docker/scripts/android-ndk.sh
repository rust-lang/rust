#!/bin/sh
set -ex

URL=https://dl.google.com/android/repository

download_ndk() {
    mkdir -p /android/ndk
    cd /android/ndk
    curl -fO $URL/$1
    unzip -q $1
    rm $1
    mv android-ndk-* ndk
}

make_standalone_toolchain() {
    # See https://developer.android.com/ndk/guides/standalone_toolchain.htm
    python3 /android/ndk/ndk/build/tools/make_standalone_toolchain.py \
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
