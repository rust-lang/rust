#!/usr/bin/env bash

set -e

bash -c "while true; do echo foo; sleep 60; done" &

if [[ "${INFINITE_SLEEP}" -eq 1 ]]; then
    echo "Sleeping for 3.5 hours y'all"
    sleep 12600
    echo "Awoke!"
elif [[ "${INFINITE_SLEEP}" -eq 2 ]]; then
    while true; do
        sleep 60000000
    done
else
    echo "travis has a lower timeout"
fi
