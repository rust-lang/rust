#!/bin/bash
# Set the SKIP_JOB environment variable if this job is supposed to only run
# when submodules are updated and they were not. The following time consuming
# tasks will be skipped when the environment variable is present.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if [[ -z "${CI_ONLY_WHEN_SUBMODULES_CHANGED+x}" ]]; then
    echo "Executing the job since there is no skip rule in effect"
elif git diff HEAD^ | grep --quiet "^index .* 160000"; then
    # Submodules pseudo-files inside git have the 160000 permissions, so when
    # those files are present in the diff a submodule was updated.
    echo "Executing the job since submodules are updated"
elif git diff --name-only HEAD^ | grep --quiet src/tools/clippy; then
    # There is not an easy blanket search for subtrees. For now, manually list
    # clippy.
    echo "Executing the job since clippy subtree was updated"
else
    echo "Not executing this job since no submodules were updated"
    ciCommandSetEnv SKIP_JOB 1
fi
