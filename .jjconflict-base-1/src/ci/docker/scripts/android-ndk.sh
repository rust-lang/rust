#!/bin/sh
set -ex

URL=https://dl.google.com/android/repository

download_ndk() {
    mkdir /android/
    cd /android
    curl -fO $URL/$1
    unzip -q $1
    rm $1
    mv android-ndk-* ndk
}
