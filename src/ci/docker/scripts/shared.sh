#!/bin/false
# shellcheck shell=bash

# This file is intended to be sourced with `. shared.sh` or
# `source shared.sh`, hence the invalid shebang and not being
# marked as an executable file in git.

function hide_output {
  { set +x; } 2>/dev/null
  on_err="
echo ERROR: An error was encountered with the build.
cat /tmp/build.log
exit 1
"
  trap "$on_err" ERR
  bash -c "while true; do sleep 30; echo \$(date) - building ...; done" &
  PING_LOOP_PID=$!
  "$@" &> /tmp/build.log
  trap - ERR
  kill $PING_LOOP_PID
  set -x
}

# See https://unix.stackexchange.com/questions/82598
# Duplicated in src/ci/shared.sh
function retry {
  echo "Attempting with retry:" "$@"
  local n=1
  local max=5
  while true; do
    "$@" && break || {
      if [[ $n -lt $max ]]; then
        sleep $n  # don't retry immediately
        ((n++))
        echo "Command failed. Attempt $n/$max:"
      else
        echo "The command has failed after $n attempts."
        return 1
      fi
    }
  done
}

download_tar_and_extract_into_dir() {
  local url="$1"
  local sum="$2"
  local dir="$3"
  local file=$(mktemp -u)

  while :; do
    if [[ -f "$file" ]]; then
      if ! h="$(sha256sum "$file" | awk '{ print $1 }')"; then
        printf 'ERROR: reading hash\n' >&2
        exit 1
      fi

      if [[ "$h" == "$sum" ]]; then
        break
      fi

      printf 'WARNING: hash mismatch: %s != expected %s\n' "$h" "$sum" >&2
      rm -f "$file"
    fi

    printf 'Downloading: %s\n' "$url"
    if ! curl -f -L -o "$file" "$url"; then
       rm -f "$file"
      sleep 1
    fi
  done

  mkdir -p "$dir"
  cd "$dir"
  tar -xf "$file"
  rm -f "$file"
}
