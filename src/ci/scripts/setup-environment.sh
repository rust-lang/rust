#!/bin/bash
# This script guesses some environment variables based on the builder name and
# the current platform, to reduce the amount of variables defined in the CI
# configuration.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

# Load extra environment variables
vars="${EXTRA_VARIABLES-}"
echo "${vars}" | jq '.' >/dev/null  # Validate JSON and exit on errors
for key in $(echo "${vars}" | jq "keys[]" -r); do
    # On Windows, for whatever reason, $key contains the BOM character in it,
    # and that messes up `jq ".${key}"`. This line strips the BOM from the key.
    #
    # https://unix.stackexchange.com/a/381263
    key="$(echo "${key}" | sed '1s/^\xEF\xBB\xBF//')"

    echo "adding extra environment variable ${key}"
    value="$(echo "${vars}" | jq ".${key}" -r)"
    export "${key}"="${value}"
    ciCommandSetEnv "${key}" "${value}"
done

# Builders starting with `dist-` are dist builders, but if they also end with
# `-alt` they are alternate dist builders.
if [[ "${CI_JOB_NAME}" = dist-* ]]; then
    if [[ "${CI_JOB_NAME}" = *-alt ]]; then
        echo "alternate dist builder detected, setting DEPLOY_ALT=1"
        ciCommandSetEnv DEPLOY_ALT 1
    else
        echo "normal dist builder detected, setting DEPLOY=1"
        ciCommandSetEnv DEPLOY 1
    fi
fi

# All the Linux builds happen inside Docker.
if isLinux; then
    if [[ -z "${IMAGE+x}" ]]; then
        echo "linux builder detected, using docker to run the build"
        ciCommandSetEnv IMAGE "${CI_JOB_NAME}"
    else
        echo "a custom docker image is already set"
    fi
fi
