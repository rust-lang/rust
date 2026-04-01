#!/bin/sh
set -ex

export ANDROID_HOME=/android/sdk
PATH=$PATH:"${ANDROID_HOME}/tools/bin"
LOCKFILE="${ANDROID_HOME}/android-sdk.lock"

# To add a new packages to the SDK or to update an existing one you need to
# run the command:
#
#    android-sdk-manager.py add-to-lockfile $LOCKFILE <package-name>
#
# Then, after every lockfile update the mirror has to be synchronized as well:
#
#    android-sdk-manager.py update-mirror $LOCKFILE
#
/scripts/android-sdk-manager.py install "${LOCKFILE}" "${ANDROID_HOME}"

details=$(cat "${LOCKFILE}" \
    | grep system-images \
    | sed 's/^system-images;android-\([0-9]\+\);default;\([a-z0-9-]\+\) /\1 \2 /g')
api="$(echo "${details}" | awk '{print($1)}')"
abi="$(echo "${details}" | awk '{print($2)}')"

# See https://developer.android.com/studio/command-line/avdmanager.html for
# usage of `avdmanager`.
echo no | avdmanager create avd \
    -n "$abi-$api" \
    -k "system-images;android-$api;default;$abi"
