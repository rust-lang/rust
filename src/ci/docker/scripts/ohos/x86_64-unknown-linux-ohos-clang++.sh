#!/bin/sh
exec /opt/ohos-sdk/native/llvm/bin/clang++ \
  -target x86_64-linux-ohos \
  --sysroot=/opt/ohos-sdk/native/sysroot \
  -D__MUSL__ \
  "$@"
