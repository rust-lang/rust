#!/bin/bash
# Set the SKIP_JOB environment variable if this job is not supposed to run on the current builder.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

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
