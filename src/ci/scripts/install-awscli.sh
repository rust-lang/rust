#!/bin/bash
# This script downloads and installs awscli from the packages mirrored in our
# own S3 bucket. This follows the recommendations at:
#
#    https://packaging.python.org/guides/index-mirrors-and-caches/#caching-with-pip
#
# To create a new mirrored copy you can run the command:
#
#    pip wheel awscli
#
# Before compressing please make sure all the wheels end with `-none-any.whl`.
# If that's not the case you'll need to remove the non-cross-platform ones and
# replace them with the .tar.gz downloaded from https://pypi.org. Also make
# sure it's possible to call this script with both Python 2 and Python 3.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

MIRROR="${MIRRORS_BASE}/2019-07-27-awscli.tar"
DEPS_DIR="/tmp/awscli-deps"
AWSCLI_SHA="537327eb34c2a495ed2c8b8a03aa926bf4ae61ce062476cdadb199d2c25d6103"

pip="pip"
pipflags=""
if isLinux; then
    pip="pip3"
    pipflags="--user"

    sudo apt-get install -y python3-setuptools python3-wheel
    ciCommandAddPath "${HOME}/.local/bin"
fi

mkdir -p "${DEPS_DIR}"
curl -O "${MIRROR}"
echo "${AWSCLI_SHA}  2019-07-27-awscli.tar" | sha256sum --check
tar xf 2019-07-27-awscli.tar -C "${DEPS_DIR}"
"${pip}" install ${pipflags} --no-index "--find-links=${DEPS_DIR}" awscli
rm -rf "${DEPS_DIR}"
