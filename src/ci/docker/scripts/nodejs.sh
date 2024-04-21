#!/bin/sh

set -ex

NODEJS_VERSION=v20.12.2
INSTALL_PATH=${1:-/node}

url="https://nodejs.org/dist/${NODEJS_VERSION}/node-${NODEJS_VERSION}-linux-x64.tar.xz"
curl -sL "$url" | tar -xJ
mv node-${NODEJS_VERSION}-linux-x64 "${INSTALL_PATH}"
