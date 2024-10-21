#!/bin/bash
# In CI environments, bootstrap is forced to use the remote upstream based
# on "git_repository" and "nightly_branch" values from src/stage0 file.
# This script configures the remote as it may not exist by default.

set -euo pipefail
IFS=$'\n\t'

ci_dir=$(cd $(dirname $0) && pwd)/..
source "$ci_dir/shared.sh"

git_repository=$(parse_stage0_file_by_key "git_repository")
nightly_branch=$(parse_stage0_file_by_key "nightly_branch")

# Configure "rust-lang/rust" upstream remote only when it's not origin.
if [ -z "$(git config remote.origin.url | grep $git_repository)" ]; then
    echo "Configuring https://github.com/$git_repository remote as upstream."
    git remote add upstream "https://github.com/$git_repository"
    REMOTE_NAME="upstream"
else
    REMOTE_NAME="origin"
fi

git fetch $REMOTE_NAME $nightly_branch
