#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

print_error() {
  echo "Error: \`$1\` is not a valid commit. To debug, run:"
  echo
  echo "    git rev-parse --verify $1"
  echo
}

full_sha() {
  git rev-parse \
    --verify \
    --quiet \
    "$1^{object}" || print_error $1
}

commit_message_with_backport_note() {
  message=$(git log --format=%B -n 1 $1)
  echo $message | awk "NR==1{print; print \"\n(backport-of: $1)\"} NR!=1"
}

cherry_pick_commit() {
  sha=$(full_sha $1)
  git cherry-pick $sha > /dev/null
  git commit \
    --amend \
    --file <(commit_message_with_backport_note $sha)
}

for arg ; do
  cherry_pick_commit $arg
done
