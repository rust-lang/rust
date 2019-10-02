#!/bin/bash -x

output=$(mktemp)

mkdir -p book/
cp -r $HOME/linkcheck/ book/
RUST_LOG=mdbook_linkcheck=debug mdbook-linkcheck -s 2>&1 | tee -a $output
cp -r book/linkcheck $HOME/

mdbook build

result=${PIPESTATUS[0]}

# if passed, great!
if [ "$result" -eq "0" ] ; then
    exit 0 ;
fi
