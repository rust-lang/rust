#!/bin/bash

set -e

./util/update_lints.py

git status --short | sort | grep -v README.md | grep -v helper.txt | sort > helper.txt

# abort if the files differ
diff "publish.files" "helper.txt"

rm helper.txt

# add all changed files
git add .
git commit -m "Bump the version"

set +e

cd clippy_lints && cargo fmt -- --write-mode=overwrite && cd ..
cargo fmt -- --write-mode=overwrite

echo "remember to add a git tag and running 'cargo test' before committing the rustfmt changes"
