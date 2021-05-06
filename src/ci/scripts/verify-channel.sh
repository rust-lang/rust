#!/bin/bash
# We want to make sure all PRs are targeting the right branch when they're
# opened, otherwise we risk (for example) to land a beta-specific change to the
# master branch. This script ensures the branch of the PR matches the channel.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

declare -A CHANNEL_BRANCH
CHANNEL_BRANCH["nightly"]="master"
CHANNEL_BRANCH["beta"]="beta"
CHANNEL_BRANCH["stable"]="stable"

if isCiBranch auto || isCiBranch try; then
    echo "channel verification is only executed on PR builds"
    exit
fi

channel=$(cat "$(ciCheckoutPath)/src/ci/channel")
branch="$(ciBaseBranch)"
if [[ "${branch}" != "${CHANNEL_BRANCH[$channel]}" ]]; then
    echo "error: PRs changing the \`${channel}\` channel should be sent to the \
\`${CHANNEL_BRANCH[$channel]}\` branch!"

    exit 1
fi
