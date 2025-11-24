#!/bin/sh

set -ex

NODEJS_VERSION=v20.12.2
YARN_VERSION=1.22.22
INSTALL_PATH=${1:-/node}

url="https://nodejs.org/dist/${NODEJS_VERSION}/node-${NODEJS_VERSION}-linux-x64.tar.xz"
curl -sL "$url" | tar -xJ
mv node-${NODEJS_VERSION}-linux-x64 "${INSTALL_PATH}"

# now, install yarn.
# we call npm through the node binary, because otherwise npm will expect node to be in the PATH
"${INSTALL_PATH}/bin/node" "${INSTALL_PATH}/bin/npm" install --global yarn@${YARN_VERSION}
