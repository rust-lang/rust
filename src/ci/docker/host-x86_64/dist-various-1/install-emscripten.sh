#!/usr/bin/env bash
set -ex

apt-get update
apt-get install -y --no-install-recommends \
  nodejs \
  default-jre

git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
