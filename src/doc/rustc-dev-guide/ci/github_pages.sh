#!/bin/bash
set -ex

BOOK_DIR=book

# Only upload the built book to github pages if it's a commit to master
if [ "$TRAVIS_BRANCH" = master -a "$TRAVIS_PULL_REQUEST" = false ]; then
    mdbook build 
    ghp-import $BOOK_DIR
fi