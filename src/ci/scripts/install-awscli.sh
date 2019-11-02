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

pip="pip"
pipflags=""
if isLinux; then
    pip="pip3"
    pipflags="--user"

    sudo apt-get install -y python3-setuptools
    echo "##vso[task.prependpath]$HOME/.local/bin"
fi

mkdir -p "${DEPS_DIR}"
curl "${MIRROR}" | tar xf - -C "${DEPS_DIR}"
"${pip}" install ${pipflags} --no-index "--find-links=${DEPS_DIR}" awscli
rm -rf "${DEPS_DIR}"
