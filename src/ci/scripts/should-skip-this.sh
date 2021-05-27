#!/bin/bash
# Set the SKIP_JOB environment variable if this job is supposed to only run
# when submodules are updated and they were not. The following time consuming
# tasks will be skipped when the environment variable is present.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if [[ -z "${CI_ONLY_WHEN_SUBMODULES_CHANGED+x}" ]]; then
    echo "Executing the job since there is no skip rule in effect"
    exit 0
fi

git fetch "https://github.com/$GITHUB_REPOSITORY" "$GITHUB_BASE_REF"
BASE_COMMIT="$(git merge-base FETCH_HEAD HEAD)"

echo "Searching for toolstate changes between $BASE_COMMIT and $(git rev-parse HEAD)"

if git diff "$BASE_COMMIT" | grep --quiet "^index .* 160000"; then
    # Submodules pseudo-files inside git have the 160000 permissions, so when
    # those files are present in the diff a submodule was updated.
    echo "Executing the job since submodules are updated"
elif ! git diff --quiet "$BASE_COMMIT" -- src/tools/clippy src/tools/rustfmt; then
    # There is not an easy blanket search for subtrees. For now, manually list
    # the subtrees.
    echo "Executing the job since clippy or rustfmt subtree was updated"
else
    echo "Not executing this job since no submodules nor subtrees were updated"
    ciCommandSetEnv SKIP_JOB 1
fi
