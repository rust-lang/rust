#!/bin/sh
exec /opt/ohos-sdk/native/llvm/bin/clang++ \
  -target arm-linux-ohos \
  --sysroot=/opt/ohos-sdk/native/sysroot \
  -D__MUSL__ \
  -march=armv7-a \
  -mfloat-abi=softfp \
  -mtune=generic-armv7-a \
  -mthumb \
  "$@"
