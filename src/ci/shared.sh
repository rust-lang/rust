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
  [ "$CI" = "true" ] || [ "$TRAVIS" = "true" ] || [ "$TF_BUILD" = "True" ]
}

function isOSX {
  [ "$TRAVIS_OS_NAME" = "osx" ] || [ "$AGENT_OS" = "Darwin" ]
}

function getCIBranch {
  if [ "$TRAVIS" = "true" ]; then
    echo "$TRAVIS_BRANCH"
  elif [ "$APPVEYOR" = "True" ]; then
    echo "$APPVEYOR_REPO_BRANCH"
  else
    echo "$BUILD_SOURCEBRANCHNAME"
  fi;
}

if ! declare -F travis_fold; then
  if [ "${TRAVIS-false}" = 'true' ]; then
    # This is a trimmed down copy of
    # https://github.com/travis-ci/travis-build/blob/master/lib/travis/build/templates/header.sh
    travis_fold() {
      echo -en "travis_fold:$1:$2\r\033[0K"
    }
    travis_time_start() {
      travis_timer_id=$(printf %08x $(( RANDOM * RANDOM )))
      travis_start_time=$(travis_nanoseconds)
      echo -en "travis_time:start:$travis_timer_id\r\033[0K"
    }
    travis_time_finish() {
      travis_end_time=$(travis_nanoseconds)
      local duration=$(($travis_end_time-$travis_start_time))
      local msg="travis_time:end:$travis_timer_id"
      echo -en "\n$msg:start=$travis_start_time,finish=$travis_end_time,duration=$duration\r\033[0K"
    }
    if [ $(uname) = 'Darwin' ]; then
      travis_nanoseconds() {
        date -u '+%s000000000'
      }
    else
      travis_nanoseconds() {
        date -u '+%s%N'
      }
    fi
  else
    travis_fold() { return 0; }
    travis_time_start() { return 0; }
    travis_time_finish() { return 0; }
  fi
fi
