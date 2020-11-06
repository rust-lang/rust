#!/bin/false
# shellcheck shell=bash

# This file is intended to be sourced with `. shared.sh` or
# `source shared.sh`, hence the invalid shebang and not being
# marked as an executable file in git.

export MIRRORS_BASE="https://ci-mirrors.rust-lang.org/rustc"

# See http://unix.stackexchange.com/questions/82598
# Duplicated in docker/dist-various-2/shared.sh
function retry {
  echo "Attempting with retry:" "$@"
  local n=1
  local max=5
  while true; do
    "$@" && break || {
      if [[ $n -lt $max ]]; then
        sleep $n  # don't retry immediately
        ((n++))
        echo "Command failed. Attempt $n/$max:"
      else
        echo "The command has failed after $n attempts."
        return 1
      fi
    }
  done
}

function isCI {
    [[ "${CI-false}" = "true" ]] || isAzurePipelines || isGitHubActions
}

function isAzurePipelines {
    [[ "${TF_BUILD-False}" = "True" ]]
}

function isGitHubActions {
    [[ "${GITHUB_ACTIONS-false}" = "true" ]]
}


function isSelfHostedGitHubActions {
    [[ "${RUST_GHA_SELF_HOSTED-false}" = "true" ]]
}

function isMacOS {
    [[ "${OSTYPE}" = "darwin"* ]]
}

function isWindows {
    [[ "${OSTYPE}" = "cygwin" ]] || [[ "${OSTYPE}" = "msys" ]]
}

function isLinux {
    [[ "${OSTYPE}" = "linux-gnu" ]]
}

function isCiBranch {
    if [[ $# -ne 1 ]]; then
        echo "usage: $0 <branch-name>"
        exit 1
    fi
    name="$1"

    if isAzurePipelines; then
        [[ "${BUILD_SOURCEBRANCHNAME}" = "${name}" ]]
    elif isGitHubActions; then
        [[ "${GITHUB_REF}" = "refs/heads/${name}" ]]
    else
        echo "isCiBranch only works inside CI!"
        exit 1
    fi
}

function ciCommit {
    if isAzurePipelines; then
        echo "${BUILD_SOURCEVERSION}"
    elif isGitHubActions; then
        echo "${GITHUB_SHA}"
    else
        echo "ciCommit only works inside CI!"
        exit 1
    fi
}

function ciCheckoutPath {
    if isAzurePipelines; then
        echo "${BUILD_SOURCESDIRECTORY}"
    elif isGitHubActions; then
        echo "${GITHUB_WORKSPACE}"
    else
        echo "ciCheckoutPath only works inside CI!"
        exit 1
    fi
}

function ciCommandAddPath {
    if [[ $# -ne 1 ]]; then
        echo "usage: $0 <path>"
        exit 1
    fi
    path="$1"

    if isAzurePipelines; then
        echo "##vso[task.prependpath]${path}"
    elif isGitHubActions; then
        echo "${path}" >> "${GITHUB_PATH}"
    else
        echo "ciCommandAddPath only works inside CI!"
        exit 1
    fi
}

function ciCommandSetEnv {
    if [[ $# -ne 2 ]]; then
        echo "usage: $0 <name> <value>"
        exit 1
    fi
    name="$1"
    value="$2"

    if isAzurePipelines; then
        echo "##vso[task.setvariable variable=${name}]${value}"
    elif isGitHubActions; then
        echo "${name}=${value}" >> "${GITHUB_ENV}"
    else
        echo "ciCommandSetEnv only works inside CI!"
        exit 1
    fi
}
