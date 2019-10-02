#!/bin/bash -x

set -o pipefail

output=$(mktemp)

mkdir -p book/
cp -r $HOME/linkcheck/ book/
RUST_LOG=mdbook_linkcheck=debug mdbook-linkcheck -s 2>&1 | tee -a $output
result=${PIPESTATUS[0]}
cp -r book/linkcheck $HOME/

mdbook build

# if passed, great!
if [ "$result" -eq "0" ] ; then
    echo "Linkchecks passed"
    exit 0 ;
else
    echo "Linkchecks failed"
    exit 1 ;
fi
