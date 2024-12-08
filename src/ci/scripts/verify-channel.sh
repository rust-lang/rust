#!/bin/bash
# We want to make sure all PRs are targeting the right branch when they're
# opened, otherwise we risk (for example) to land a beta-specific change to the
# master branch. This script ensures the branch of the PR matches the channel.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isCiBranch auto || isCiBranch try || isCiBranch try-perf || isCiBranch automation/bors/try; then
    echo "channel verification is only executed on PR builds"
    exit
fi

channel=$(cat "$(ciCheckoutPath)/src/ci/channel")
case "${channel}" in
    nightly)
        channel_branch="master"
        ;;
    beta)
        channel_branch="beta"
        ;;
    stable)
        channel_branch="stable"
        ;;
    *)
        echo "error: unknown channel defined in src/ci/channel: ${channel}"
        exit 1
esac

branch="$(ciBaseBranch)"
if [[ "${branch}" != "${channel_branch}" ]]; then
    echo "error: PRs changing the \`${channel}\` channel should be sent to the \
\`${channel_branch}\` branch!"

    exit 1
fi
