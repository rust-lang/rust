#!/usr/bin/env bash

set -ev
set -o pipefail

# https://docs.travis-ci.com/user/environment-variables/#default-environment-variables
if [ "$TRAVIS_EVENT_TYPE" = "cron" ] ; then # running in cron job
  FLAGS=""

  echo "Doing full link check."
elif [ "$CI" = "true" ] ; then # running in PR CI build
  if [ -z "$TRAVIS_COMMIT_RANGE" ]; then
    echo "error: unexpected state: TRAVIS_COMMIT_RANGE must be non-empty in CI"
    exit 1
  fi

  CHANGED_FILES=$(git diff --name-only $TRAVIS_COMMIT_RANGE | tr '\n' ' ')
  FLAGS="--no-cache -f $CHANGED_FILES"

  echo "Checking files changed in $TRAVIS_COMMIT_RANGE: $CHANGED_FILES"
else # running locally
  COMMIT_RANGE=master...
  CHANGED_FILES=$(git diff --name-only $COMMIT_RANGE | tr '\n' ' ')
  FLAGS="-f $CHANGED_FILES"

  echo "Checking files changed in $COMMIT_RANGE: $CHANGED_FILES"
fi

exec mdbook-linkcheck $FLAGS -- $TRAVIS_BUILD_DIR
