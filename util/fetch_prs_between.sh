#!/bin/bash

# Fetches the merge commits between two git commits and prints the PR URL
# together with the full commit message
#
# If you want to use this to update the Clippy changelog, be sure to manually
# exclude the non-user facing changes like 'rustup' PRs, typo fixes, etc.

first=$1
last=$2

IFS='
'
for pr in $(git log --oneline --grep "Merge #" --grep "Merge pull request" --grep "Auto merge of" --grep "Rollup merge of" "$first...$last" | sort -rn | uniq); do
  id=$(echo "$pr" | rg -o '#[0-9]{3,5}' | cut -c 2-)
  commit=$(echo "$pr" | cut -d' ' -f 1)
  message=$(git --no-pager show --pretty=medium "$commit")
  if [[ -n $(echo "$message" | rg "^[\s]{4}changelog: [nN]one\.*$") ]]; then
    continue
  fi

  echo "URL: https://github.com/rust-lang/rust-clippy/pull/$id"
  echo "Markdown URL: [#$id](https://github.com/rust-lang/rust-clippy/pull/$id)"
  echo "$message"
  echo "---------------------------------------------------------"
  echo
done
