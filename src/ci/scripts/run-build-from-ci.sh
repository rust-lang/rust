#!/bin/bash
# Start the CI build. You shouldn't run this locally: call either src/ci/run.sh
# or src/ci/docker/run.sh instead.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

export CI="true"
export SRC=.

# Remove any preexisting rustup installation since it can interfere
# with the cargotest step and its auto-detection of things like Clippy in
# the environment
rustup self uninstall -y || true
if [ -z "${IMAGE+x}" ]; then
    if ! src/ci/run.sh; then
        cat build/x86_64-unknown-linux-gnu/test/codegen/issue-75742-format_without_fmt_args/*.ll
    fi
else
    src/ci/docker/run.sh "${IMAGE}"
fi
