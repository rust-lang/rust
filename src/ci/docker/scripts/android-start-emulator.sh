#!/bin/sh

set -ex

# Setting SHELL to a file instead on a symlink helps android
# emulator identify the system
export SHELL=/bin/bash

nohup nohup emulator @$1-21 \
    -engine $2 -no-window -no-audio -partition-size 2047 0<&- &>/dev/null &

shift 2

exec "$@"
