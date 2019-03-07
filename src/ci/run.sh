#!/usr/bin/env bash

set -e

if [[ "${INFINITE_SLEEP}" -eq 1 ]]; then
    if ! [[ "${INSTALL_STEP}" -eq 1 ]]; then
        echo "Sleeping for 200 minutes y'all"
        i=0
        while [[ "$i" -le 200 ]]; do
            echo "$i minutes passed"
            i=$((i+1))
            sleep 60
        done
        echo "Awoke!"
    else
        echo "skipping timeout on the install phase"
    fi
elif [[ "${INFINITE_SLEEP}" -eq 2 ]]; then
    echo "Sleeping for 100 minutes y'all"
    i=0
    while [[ "$i" -le 100 ]]; do
        echo "$i minutes passed"
        i=$((i+1))
        sleep 60
    done
    echo "Awoke!"
elif [[ "${INFINITE_SLEEP}" -eq 3 ]]; then
    while true; do
        echo "foo"
        sleep 60
    done
else
    echo "travis has a lower timeout"
fi
