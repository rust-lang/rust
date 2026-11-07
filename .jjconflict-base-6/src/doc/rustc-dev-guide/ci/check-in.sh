#!/bin/sh

set -eu

# This is not a very smart script
if [ $# != 2 ]; then
  echo "usage: $0 <since> <number-of-prs-merged>"
  if [ $# = 0 ]; then
    echo "help: you can find the last check-in at" \
         "https://rust-lang.zulipchat.com/#narrow/stream/238009-t-compiler.2Fmeetings/search/wg-rustc-dev-guide"
  elif [ $# = 1 ] ; then
    echo "help: you can find the number of PRs merged at" \
         "https://github.com/rust-lang/rustc-dev-guide/pulls?q=is%3Apr+is%3Amerged+updated%3A%3E$1"
  fi
  exit 1
fi

curl() {
  command curl -s "$@"
}

# Get recently updated PRs
curl "https://api.github.com/repos/rust-lang/rustc-dev-guide/pulls?state=closed&per_page=$2" \
  | jq '[.[] | select(.merged_at > "'"$1"'")]' > pulls.json

show_pulls() {
  jq -r '.[] | { title, number, html_url, user: .user.login } | "- " + .title + " [#" + (.number | tostring) + "](" + .html_url + ")"'
}

echo "### Most notable changes"
echo
show_pulls < pulls.json
echo
echo "### Most notable WIPs"
echo
# If there are more than 30 PRs open at a time, you'll need to set `per_page`.
# For now this seems unlikely.
curl "https://api.github.com/repos/rust-lang/rustc-dev-guide/pulls?state=open" | show_pulls
