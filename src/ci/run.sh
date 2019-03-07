#!/usr/bin/env bash

set -e

if [[ "${INFINITE_SLEEP}" -eq 1 ]]; then
    echo "Sleeping for 3.5 hours y'all"
    i=0
    while [[ "$i" -le 210 ]]; do
        echo "$i minutes passed"
        i=$((i+1))
        sleep 60
    done
    echo "Awoke!"
elif [[ "${INFINITE_SLEEP}" -eq 2 ]]; then
    while true; do
        echo "foo"
        sleep 60
    done
else
    echo "travis has a lower timeout"
fi
