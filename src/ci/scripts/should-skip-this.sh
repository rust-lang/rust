#!/bin/bash
# Set the SKIP_JOB environment variable if this job is supposed to only run
# when submodules are updated and they were not. The following time consuming
# tasks will be skipped when the environment variable is present.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if [[ -n "${CI_ONLY_WHEN_SUBMODULES_CHANGED-}" ]]; then
    git fetch "https://github.com/$GITHUB_REPOSITORY" "$GITHUB_BASE_REF"
    BASE_COMMIT="$(git merge-base FETCH_HEAD HEAD)"

    echo "Searching for toolstate changes between $BASE_COMMIT and $(git rev-parse HEAD)"

    if git diff "$BASE_COMMIT" | grep --quiet "^index .* 160000"; then
        # Submodules pseudo-files inside git have the 160000 permissions, so when
        # those files are present in the diff a submodule was updated.
        echo "Submodules were updated"
    elif ! git diff --quiet "$BASE_COMMIT" -- src/tools/clippy src/tools/rustfmt; then
        # There is not an easy blanket search for subtrees. For now, manually list
        # the subtrees.
        echo "Clippy or rustfmt subtrees were updated"
    elif ! (git diff --quiet "$BASE_COMMIT" -- \
             src/test/rustdoc-gui \
             src/librustdoc \
             src/ci/docker/host-x86_64/x86_64-gnu-tools/Dockerfile \
             src/tools/rustdoc-gui); then
        # There was a change in either rustdoc or in its GUI tests.
        echo "Rustdoc was updated"
    else
        echo "Not executing this job since no submodules nor subtrees were updated"
        ciCommandSetEnv SKIP_JOB 1
        exit 0
    fi
fi

if [[ -n "${CI_ONLY_WHEN_CHANNEL-}" ]]; then
    if [[ "${CI_ONLY_WHEN_CHANNEL}" = "$(cat src/ci/channel)" ]]; then
        echo "The channel is the expected one"
    else
        echo "Not executing this job as the channel is not the expected one"
        ciCommandSetEnv SKIP_JOB 1
        exit 0
    fi
fi


echo "Executing the job since there is no skip rule preventing the execution"
exit 0
