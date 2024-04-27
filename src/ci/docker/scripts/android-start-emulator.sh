#!/bin/sh

set -ex

# Setting SHELL to a file instead on a symlink helps android
# emulator identify the system
export SHELL=/bin/bash

# Using the default qemu2 engine makes time::tests::since_epoch fails because
# the emulator date is set to unix epoch (in armeabi-v7a-18 image). Using
# classic engine the emulator starts with the current date and the tests run
# fine. If another image is used, this need to be evaluated again.
nohup nohup emulator @armeabi-v7a-18 \
    -engine classic -no-window -partition-size 2047 0<&- &>/dev/null &

exec "$@"
