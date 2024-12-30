#!/usr/bin/env bash

set -e
set -o pipefail

LINKCHECK_BINARY=mdbook-linkcheck2

set_github_token() {
  jq '.config.output.linkcheck."http-headers"."github\\.com" = ["Authorization: Bearer $GITHUB_TOKEN"]'
}

if [ -z "$ENABLE_LINKCHECK" ] ; then
  echo "Skipping link check."
  exit 0
fi

# https://docs.github.com/en/actions/reference/environment-variables
if [ "$GITHUB_EVENT_NAME" = "schedule" ] ; then # running in scheduled job
  FLAGS=""
  USE_TOKEN=1

  echo "Doing full link check."
elif [ "$GITHUB_EVENT_NAME" = "pull_request" ] ; then # running in PR CI build
  if [ -z "$BASE_SHA" ]; then
    echo "error: unexpected state: BASE_SHA must be non-empty in CI"
    exit 1
  fi

  CHANGED_FILES=$(git diff --name-only $BASE_SHA... | sed 's#^src/##' | tr '\n' ' ')
  FLAGS="--no-cache -f $CHANGED_FILES"
  USE_TOKEN=1

  echo "Checking files changed since $BASE_SHA: $CHANGED_FILES"
else # running locally
  COMMIT_RANGE=master...
  CHANGED_FILES=$(git diff --name-only $COMMIT_RANGE | sed 's#^src/##' | tr '\n' ' ')
  FLAGS="-f $CHANGED_FILES"

  echo "Checking files changed in $COMMIT_RANGE: $CHANGED_FILES"
fi

echo "exec $LINKCHECK_BINARY $FLAGS"
if [ "$USE_TOKEN" = 1 ]; then
  config=$(set_github_token)
  exec $LINKCHECK_BINARY $FLAGS <<<"$config"
else
  exec $LINKCHECK_BINARY $FLAGS
fi
