#!/bin/bash
# Start the CI build. You shouldn't run this locally: call either src/ci/run.sh
# or src/ci/docker/run.sh instead.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

export CI="true"
export SRC=.

echo "::add-matcher::src/ci/github-actions/problem_matchers.json"

# Remove any preexisting rustup installation since it can interfere
# with the cargotest step and its auto-detection of things like Clippy in
# the environment
rustup self uninstall -y || true
if [ -z "${IMAGE+x}" ]; then
    src/ci/run.sh
else
    src/ci/docker/run.sh "${IMAGE}"
fi
