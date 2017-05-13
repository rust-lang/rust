#!/bin/sh

# Builds documentation for all target triples that we have a registered URL for
# in liblibc. This scrapes the list of triples to document from `src/lib.rs`
# which has a bunch of `html_root_url` directives we pick up.

set -e

TARGETS=`grep html_root_url src/lib.rs | sed 's/.*".*\/\(.*\)"/\1/'`

rm -rf target/doc
mkdir -p target/doc

cp ci/landing-page-head.html target/doc/index.html

for target in $TARGETS; do
  echo documenting $target

  rustdoc -o target/doc/$target --target $target src/lib.rs --cfg dox \
    --crate-name libc

  echo "<li><a href="/libc/$target/libc/index.html">$target</a></li>" \
    >> target/doc/index.html
done

cat ci/landing-page-footer.html >> target/doc/index.html

# If we're on travis, not a PR, and on the right branch, publish!
if [ "$TRAVIS_PULL_REQUEST" = "false" ] && [ "$TRAVIS_BRANCH" = "master" ]; then
  pip install ghp-import --user $USER
  $HOME/.local/bin/ghp-import -n target/doc
  git push -qf https://${GH_TOKEN}@github.com/${TRAVIS_REPO_SLUG}.git gh-pages
fi
