#!/bin/false

# This file is intended to be sourced with `. shared.sh` or
# `source shared.sh`, hence the invalid shebang and not being
# marked as an executable file in git.

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

function isOSX {
  [ "$AGENT_OS" = "Darwin" ]
}

function getCIBranch {
  echo "$BUILD_SOURCEBRANCHNAME"
}
