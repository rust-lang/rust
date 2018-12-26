#!/bin/bash

set -eux

# Install prerequisites.
apt-get update
apt-get install -y --no-install-recommends \
  apt-transport-https \
  ca-certificates \
  clang-5.0 \
  cmake \
  curl \
  file \
  g++ \
  gdb \
  git \
  lld-5.0 \
  make \
  python \
  sudo \
  xz-utils

# Set up a Clang-based cross compiler toolchain.
# Based on the steps described at https://nuxi.nl/cloudabi/debian/
target=$1
for tool in ar nm objdump ranlib size; do
  ln -s ../lib/llvm-5.0/bin/llvm-${tool} /usr/bin/${target}-${tool}
done
ln -s ../lib/llvm-5.0/bin/clang /usr/bin/${target}-cc
ln -s ../lib/llvm-5.0/bin/clang /usr/bin/${target}-c++
ln -s ../lib/llvm-5.0/bin/lld /usr/bin/${target}-ld
ln -s ../../${target} /usr/lib/llvm-5.0/${target}

# Install the C++ runtime libraries from CloudABI Ports.
echo deb https://nuxi.nl/distfiles/cloudabi-ports/debian/ cloudabi cloudabi > \
    /etc/apt/sources.list.d/cloudabi.list
curl 'https://pgp.mit.edu/pks/lookup?op=get&search=0x0DA51B8531344B15' | \
    apt-key add -
apt-get update
apt-get install -y $(echo ${target} | sed -e s/_/-/g)-cxx-runtime
