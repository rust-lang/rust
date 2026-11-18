#!/bin/bash

# Publish the backtrace crate to crates.io
# Should be called from a CI workflow
# Expects the following environment variables as input:
# - GITHUB_SHA: SHA of the currently merged commit from which backtrace should be published
# - TEMPDIR: scratch directory for temporary files

set -euxo pipefail

# Find PR for the current commit
HEAD_PR=$(gh pr list --search "${GITHUB_SHA}" --state merged --json number --jq '.[0].number')
if [ -z "${HEAD_PR}" ]; then
    echo "PR for commit SHA ${GITHUB_SHA} not found, exiting"
    exit 1
fi
echo "HEAD: ${GITHUB_SHA} (#${HEAD_PR})"

git submodule update --init library/backtrace
cd library/backtrace

# Find the version of backtrace
VERSION=$(
    cargo metadata --no-deps --format-version 1 |
    jq -r '.packages[] | select(.name == "backtrace") | .version'
)
CRATE=backtrace

# Check if the version already exists on crates.io with a HEAD request to the CDN
code=$(
    curl --head --silent -o /dev/null --write-out '%{http_code}' \
    "https://static.crates.io/crates/${CRATE}/${CRATE}-${VERSION}.crate"
)

# If the crate version already exists, then there is nothing left to do
if [[ $code == 200 ]]; then
    exit 0
fi

set +e
cargo publish --locked &> "${TEMPDIR}"/cargo-log.txt
STATUS=$?
set -e

COMMENT_FILE=${TEMPDIR}/comment.txt

if [[ ${STATUS} == 0 ]]; then
    echo "backtrace \`${VERSION}\` published to crates.io" > "${COMMENT_FILE}"
else
    echo "backtrace \`${VERSION}\` could not be published to crates.io" > "${COMMENT_FILE}"
    printf "<details>\n<summary>Cargo publish log</summary>\n\n" >> "${COMMENT_FILE}"
    printf '```\n' >> "${COMMENT_FILE}"
    cat "${TEMPDIR}/cargo-log.txt" >> "${COMMENT_FILE}"
    printf '\n```\n' >> "${COMMENT_FILE}"
    printf "</details>\n\n" >> "${COMMENT_FILE}"
fi

# Comment on the PR to let it know how it went
gh pr comment "${HEAD_PR}" -F "${COMMENT_FILE}"
exit "${STATUS}"
