#!/bin/bash
# On the stable channel, check whether we're trying to build artifacts with the
# same version number of a release that's already been published, and fail the
# build if that's the case.
#
# It's a mistake whenever that happens: the release process won't start if it
# detects a duplicate version number, and the artifacts would have to be
# rebuilt anyway.

set -euo pipefail
IFS=$'\n\t'

if [[ "$(cat src/ci/channel)" != "stable" ]]; then
    echo "This script only works on the stable channel. Skipping the check."
    exit 0
fi

version="$(cat src/version)"
url="https://static.rust-lang.org/dist/channel-rust-${version}.toml"

if curl --silent --fail "${url}" >/dev/null; then
    echo "The version number ${version} matches an existing release."
    echo
    echo "If you're trying to prepare a point release, remember to change the"
    echo "version number in the src/version file."
    exit 1
else
    echo "The version number ${version} does not match any released version!"
    exit 0
fi
