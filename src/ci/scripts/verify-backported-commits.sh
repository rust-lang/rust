#!/bin/bash
# Ensure commits in beta are in master & commits in stable are in beta + master.
set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

# We don't care about commits that predate this automation check, so we pass a
# `<limit>` argument to `git cherry`.
BETA_LIMIT="53fd98ca776cb875bc9e5514f56b52eb74f9e7a9"
STABLE_LIMIT="a178d0322ce20e33eac124758e837cbd80a6f633"

verify_backported_commits_main() {
  ci_base_branch=$(ciBaseBranch)

  if [[ "$ci_base_branch" != "beta" && "$ci_base_branch" != "stable" ]]; then
    echo 'Skipping. This is only run when merging to the beta or stable branches.'
    exit 0
  fi

  if [[ $ci_base_branch == "beta" ]]; then
    verify_cherries master "$BETA_LIMIT" \
      || exit 1

  elif [[ $ci_base_branch == "stable" ]]; then
    (verify_cherries master "$STABLE_LIMIT" \
      & verify_cherries beta "$STABLE_LIMIT") \
      || exit 1

  fi
}

# Verify all commits in `HEAD` are backports of a commit in <upstream>. See
# https://git-scm.com/docs/git-cherry for an explanation of the arguments.
#
# $1 = <upstream>
# $2 = <limit>
verify_cherries() {
  # commits that lack a `backport-of` comment.
  local no_backports=()
  # commits with an incorrect `backport-of` comment.
  local bad_backports=()

  commits=$(git cherry "origin/$1" HEAD "$2")

  if [[ -z "$commits" ]]; then
    echo "All commits in \`HEAD\` are present in \`$1\`"
    return 0
  fi

  commits=$(echo "$commits" | grep '^\+' | cut -c 3-)

  while read sha; do
    # Check each commit in <current>..<upstream>
    backport_sha=$(get_backport "$sha")

    if [[ "$backport_sha" == "nothing" ]]; then
      echo "✓ \`$sha\` backports nothing"
      continue
    fi

    if [[ -z "$backport_sha" ]]; then
      no_backports+=("$sha")
      continue
    fi

    if ! is_in_master "$backport_sha"; then
      bad_backports+=("$sha")
      continue
    fi

    echo "✓ \`$sha\` backports \`$backport_sha\`"
  done <<< "$commits"

  failure=0

  if [ ${#no_backports[@]} -ne 0 ]; then
        echo 'Error: Could not find backports for all commits.'
        echo
        echo 'All commits in \`HEAD\` are required to have a corresponding upstream commit.'
        echo 'It looks like the following commits:'
        echo
        for commit in "${no_backports[@]}"; do
          echo "    $commit"
        done
        echo
        echo "do not match any commits in \`$1\`. If this was intended, add the text"
        echo '\`backport-of: <SHA of a commit already in master>\`'
        echo 'somewhere in the message of each of these commits.'
        echo
        failure=1
  fi

  if [ ${#bad_backports[@]} -ne 0 ]; then
        echo 'Error: Found incorrectly marked commits.'
        echo
        echo 'The following commits:'
        echo
        for commit in "${bad_backports[@]}"; do
          echo "    $commit"
        done
        echo
        echo 'have commit messages marked \`backport-of: <SHA>\`, but the SHA is not in'
        echo '\`master\`.'
        echo
        failure=1
  fi

  return $failure
}

# Get the backport of a commit. It echoes one of:
#
# 1. A SHA of the backported commit
# 2. The string "nothing"
# 3. An empty string
#
# $1 = <sha>
get_backport() {
  # This regex is:
  #
  # ^.* - throw away any extra starting characters
  # backport-of: - prefix
  # \s\? - optional space
  # \(\) - capture group
  # [a-f0-9]\+\|nothing - a SHA or the text 'nothing'
  # .* - throw away any extra ending characters
  # \1 - replace it with the first match
  # {s//\1/p;q} - print the first occurrence and quit
  #
  git show -s --format=%B "$1" \
    | sed -n '/^.*backport-of:\s\?\([a-f0-9]\+\|nothing\).*/{s//\1/p;q}'
}

# Check if a commit is in master.
#
# $1 = <sha>
is_in_master() {
  git merge-base --is-ancestor "$1" origin/master 2> /dev/null
}

verify_backported_commits_main
