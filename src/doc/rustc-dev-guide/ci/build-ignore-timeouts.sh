#!/bin/bash -x

output=$(mktemp)

RUST_LOG=mdbook_linkcheck=debug mdbook build 2>&1 | tee $output

result=${PIPESTATUS[0]}

# if passed, great!
if [ "$result" -eq "0" ] ; then
    exit 0 ;
fi

errors=$(cat $output | sed -n 's/There \(was\|were\) \([0-9]\+\).*$/\2/p')
timeouts=$(cat $output | grep "error while fetching" | wc -l)

# if all errors are timeouts, ignore them...
if [ "$errors" -eq "$timeouts" ] ; then
    echo "Ignoring $timeouts timeouts";
    exit 0;
else
    echo "Non-timeout errors found";
    exit 1;
fi
