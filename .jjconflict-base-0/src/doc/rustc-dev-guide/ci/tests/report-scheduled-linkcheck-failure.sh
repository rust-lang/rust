#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
script="$repo_root/ci/report-scheduled-linkcheck-failure.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

export PATH="$repo_root/ci/tests/fakes:$PATH"
export GH_MOCK_LOG="$tmp/gh.log"
export GITHUB_REPOSITORY="rust-lang/rustc-dev-guide"
export GITHUB_RUN_ID=123
export GITHUB_RUN_NUMBER=456
export GITHUB_SERVER_URL="https://github.com"
export GITHUB_SHA=0123456789abcdef
export GH_TOKEN=test-token

fail() {
  echo "error: $*" >&2
  exit 1
}

assert_log_contains() {
  grep -F -- "$1" "$GH_MOCK_LOG" >/dev/null || fail "gh log does not contain: $1"
}

assert_log_excludes() {
  if grep -F -- "$1" "$GH_MOCK_LOG" >/dev/null; then
    fail "gh log unexpectedly contains: $1"
  fi
}

# A first failure creates the issue with all labels and links to the failed job.
: > "$GH_MOCK_LOG"
unset GH_MOCK_ISSUE_NUMBER GH_MOCK_API_FAIL
"$script" >/dev/null
assert_log_contains 'issue <list> <--repo> <rust-lang/rustc-dev-guide> <--state> <open>'
assert_log_contains '<--label> <C-broken-links> <--label> <C-CI> <--label> <A-linkcheck>'
assert_log_contains '<--json> <number,title> <--jq> <[.[] | select(.title == "[automation] Dead links found")][0].number // empty>'
assert_log_contains 'issue <create>'
assert_log_contains '<--title> <[automation] Dead links found>'
assert_log_contains '<--label> <C-broken-links> <--label> <C-CI> <--label> <A-linkcheck>'
assert_log_contains '[CI run #456](https://github.example/jobs/123)'
assert_log_excludes 'issue <comment>'

# A later failure comments on the existing issue instead of creating another.
: > "$GH_MOCK_LOG"
export GH_MOCK_ISSUE_NUMBER=42
"$script" >/dev/null
assert_log_contains 'issue <comment> <42>'
assert_log_contains 'failed again in [CI run #456](https://github.example/jobs/123)'
assert_log_excludes 'issue <create>'

# GitHub API errors and missing required environment variables remain fatal.
: > "$GH_MOCK_LOG"
export GH_MOCK_API_FAIL=1
if "$script" >/dev/null 2>&1; then
  fail "script succeeded after gh api failed"
fi
unset GH_MOCK_API_FAIL

for variable in GITHUB_REPOSITORY GITHUB_RUN_ID GITHUB_RUN_NUMBER GITHUB_SERVER_URL GITHUB_SHA GH_TOKEN; do
  if env -u "$variable" "$script" >/dev/null 2>&1; then
    fail "script succeeded without $variable"
  fi
done

echo "report-scheduled-linkcheck-failure tests passed"
