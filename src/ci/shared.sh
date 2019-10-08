#!/bin/false

# This file is intended to be sourced with `. shared.sh` or
# `source shared.sh`, hence the invalid shebang and not being
# marked as an executable file in git.

export MIRRORS_BASE="https://rust-lang-ci-mirrors.s3-us-west-1.amazonaws.com/rustc"

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
  [ "$CI" = "true" ] || [ "$TF_BUILD" = "True" ]
}

function isMacOS {
  [ "$AGENT_OS" = "Darwin" ]
}

function isWindows {
  [ "$AGENT_OS" = "Windows_NT" ]
}

function isLinux {
  [ "$AGENT_OS" = "Linux" ]
}

function getCIBranch {
  echo "$BUILD_SOURCEBRANCHNAME"
}

function ciCommandAddPath {
    if [[ $# -ne 1 ]]; then
        echo "usage: $0 <path>"
        exit 1
    fi
    path="$1"

    echo "##vso[task.prependpath]${path}"
}

function ciCommandSetEnv {
    if [[ $# -ne 2 ]]; then
        echo "usage: $0 <name> <value>"
        exit 1
    fi
    name="$1"
    value="$2"

    echo "##vso[task.setvariable variable=${name}]${value}"
}
