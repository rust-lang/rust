#!/bin/bash

# Fetches the merge commits between two git commits and prints the PR URL
# together with the full commit message
#
# If you want to use this to update the Clippy changelog, be sure to manually
# exclude the non-user facing changes like 'rustup' PRs, typo fixes, etc.

set -e

IFS='
'
for pr in $(git log --oneline --merges --first-parent "$1...$2"); do
  id=$(echo "$pr" | rg -o '#[0-9]{3,5}' | cut -c 2-)
  commit=$(echo "$pr" | cut -d' ' -f 1)
  message=$(git --no-pager show --pretty=medium "$commit")

  if [[ -z "$newest_pr" ]]; then
    newest_pr="$id"
  fi
  oldest_pr="$id"

  if [[ -n $(echo "$message" | rg "^[\s]{4}changelog: [nN]one\.*$") ]]; then
    continue
  fi

  echo "URL: https://github.com/rust-lang/rust-clippy/pull/$id"
  echo "Markdown URL: [#$id](https://github.com/rust-lang/rust-clippy/pull/$id)"
  echo "$message"
  echo "---------------------------------------------------------"
  echo
done

newest_merged_at="$(gh pr view -R rust-lang/rust-clippy --json mergedAt $newest_pr -q .mergedAt)"
oldest_merged_at="$(gh pr view -R rust-lang/rust-clippy --json mergedAt $oldest_pr -q .mergedAt)"

query="merged:$oldest_merged_at..$newest_merged_at base:master"
encoded_query="$(echo $query | sed 's/ /+/g; s/:/%3A/g')"

pr_link="https://github.com/rust-lang/rust-clippy/pulls?q=$encoded_query"
count="$(gh api -X GET search/issues -f "q=$query repo:rust-lang/rust-clippy" -q .total_count)"

echo "[View all $count merged pull requests]($pr_link)"
