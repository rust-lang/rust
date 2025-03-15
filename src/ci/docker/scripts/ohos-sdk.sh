#!/bin/sh
set -ex

URL=https://repo.huaweicloud.com/openharmony/os/5.0.0-Release/ohos-sdk-windows_linux-public.tar.gz

curl $URL | tar xz -C /tmp linux/native-linux-x64-5.0.0.71-Release.zip
mkdir /opt/ohos-sdk
cd /opt/ohos-sdk
unzip -qq /tmp/linux/native-linux-x64-5.0.0.71-Release.zip
rm /tmp/linux/native-linux-x64-5.0.0.71-Release.zip
