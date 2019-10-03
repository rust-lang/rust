#!/bin/bash -x

set -e

mkdir -p book/
cp -r $HOME/linkcheck/ book/
RUST_LOG=mdbook_linkcheck=debug mdbook-linkcheck -s
cp -r book/linkcheck $HOME/

mdbook build
