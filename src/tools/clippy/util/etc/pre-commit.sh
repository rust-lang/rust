#!/bin/sh

# hide output
set -e

# Update lints
cargo dev update_lints
git add clippy_lints/src/lib.rs

# Formatting:
#     Git will not automatically add the formatted code to the staged changes once
#     fmt was executed. This collects all staged files rs files that are currently staged.
#     They will later be added back.
#
#     This was proudly stolen and adjusted from here:
#     https://medium.com/@harshitbangar/automatic-code-formatting-with-git-66c3c5c26798
files=$( (git diff --cached --name-only --diff-filter=ACMR | grep -Ei "\.rs$") || true)
if [ ! -z "${files}" ]; then
    cargo dev fmt
    git add $(echo "$files" | paste -s -d " " -)
fi
