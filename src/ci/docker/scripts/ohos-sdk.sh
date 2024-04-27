#!/bin/sh
set -ex

URL=https://repo.huaweicloud.com/openharmony/os/4.0-Release/ohos-sdk-windows_linux-public.tar.gz

curl $URL | tar xz -C /tmp ohos-sdk/linux/native-linux-x64-4.0.10.13-Release.zip
mkdir /opt/ohos-sdk
cd /opt/ohos-sdk
unzip -qq /tmp/ohos-sdk/linux/native-linux-x64-4.0.10.13-Release.zip
