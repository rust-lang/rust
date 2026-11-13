#!/usr/bin/env bash

set -euo pipefail

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID must be set}"
: "${GITHUB_RUN_NUMBER:?GITHUB_RUN_NUMBER must be set}"
: "${GITHUB_SERVER_URL:?GITHUB_SERVER_URL must be set}"
: "${GITHUB_SHA:?GITHUB_SHA must be set}"
: "${GH_TOKEN:?GH_TOKEN must be set}"

title="[automation] Dead links found"
job_url=$(gh api \
  "repos/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID/jobs" \
  --jq '.jobs[] | select(.name == "ci") | .html_url')

issue_number=$(gh issue list \
  --repo "$GITHUB_REPOSITORY" \
  --state open \
  --label C-broken-links \
  --label C-CI \
  --label A-linkcheck \
  --limit 100 \
  --json number,title \
  --jq '[.[] | select(.title == "[automation] Dead links found")][0].number // empty')

if [[ -n "$issue_number" ]]; then
  gh issue comment "$issue_number" \
    --repo "$GITHUB_REPOSITORY" \
    --body "The scheduled link check failed again in [CI run #$GITHUB_RUN_NUMBER]($job_url)."
  exit 0
fi

body=$(cat <<EOF
The scheduled link check failed in [CI run #$GITHUB_RUN_NUMBER]($job_url).

Commit: [$GITHUB_SHA]($GITHUB_SERVER_URL/$GITHUB_REPOSITORY/commit/$GITHUB_SHA)

Please inspect the failed run for broken links or an automation error. Close this issue after the scheduled link check succeeds again.

**WARNING**: Do not rename this issue or remove \`C-broken-links\`, \`C-CI\`, or \`A-linkcheck\`; doing so will break the automation.
EOF
)

gh issue create \
  --repo "$GITHUB_REPOSITORY" \
  --title "$title" \
  --body "$body" \
  --label C-broken-links \
  --label C-CI \
  --label A-linkcheck
