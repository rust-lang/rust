#!/bin/bash
# Compress doc artifacts and name them based on the commit, or the date if
# commit is not available.

set -euox pipefail

# Try to get short commit hash, fallback to date
if [ -n "$HEAD_SHA" ]; then
    short_rev=$(echo "${HEAD_SHA}" | cut -c1-8)
else
    short_rev=$(git rev-parse --short HEAD || date -u +'%Y-%m-%dT%H%M%SZ')
fi

# Try to get branch, fallback to none
branch=$(git branch --show-current || echo)

if [ -n "$branch" ]; then
    branch="${branch}-"
fi

if [ "${GITHUB_EVENT_NAME:=none}" = "pull_request" ]; then
    pr_num=$(echo "$GITHUB_REF_NAME" | cut -d'/' -f1)
    name="doc-${pr_num}-${short_rev}"
else
    name="doc-${branch}${short_rev}"
fi


if [ -d "obj/staging/doc" ]; then
    mkdir -p obj/artifacts/doc

    # Level 12 seems to give a good tradeoff of time vs. space savings
    ZSTD_CLEVEL=12 ZSTD_NBTHREADS=4 \
    tar --zstd -cf "obj/artifacts/doc/${name}.tar.zst" -C obj/staging/doc .

    ls -lh obj/artifacts/doc
fi

# Set this environment variable for future use if running in CI
if [ -n "$GITHUB_ENV" ]; then
    echo "DOC_ARTIFACT_NAME=${name}" >> "$GITHUB_ENV"
fi
