#!/bin/bash

# Small script to run tests for a target (or all targets) inside all the
# respective docker images.

set -euxo pipefail

run() {
    local target="$1"

    echo "TESTING TARGET: $target"

    # This directory needs to exist before calling docker, otherwise docker will create it but it
    # will be owned by root
    mkdir -p target

    if [ $(uname -s) = "Linux" ] && [ -z "${DOCKER_BASE_IMAGE:-}" ]; then
      # Share the host rustc and target. Do this only on Linux and if the image
      # isn't overridden
      run_args=(
           --user "$(id -u):$(id -g)"
           -e "CARGO_HOME=/cargo"
           -v "${HOME}/.cargo:/cargo"
           -v "$(pwd)/target:/builtins-target" 
           -v "$(rustc --print sysroot):/rust:ro"
      )
      run_cmd="HOME=/tmp PATH=\$PATH:/rust/bin ci/run.sh $target"
    else
      # Use rustc provided by a docker image
      docker volume create compiler-builtins-cache
      build_args=(
        "--build-arg" "IMAGE=${DOCKER_BASE_IMAGE:-rustlang/rust:nightly}"
      )
      run_args=(
        -v "compiler-builtins-cache:/builtins-target"
      )
      run_cmd="HOME=/tmp USING_CONTAINER_RUSTC=1 ci/run.sh $target"
    fi

    if [ -d compiler-rt ]; then
      export RUST_COMPILER_RT_ROOT=./compiler-rt
    fi

    if [ "$GITHUB_ACTIONS" = "true" ]; then
      # Enable Docker image caching on GHA
      
      buildx="buildx"
      build_args=(
        "--cache-from" "type=local,src=/tmp/.buildx-cache"
        "--cache-to" "type=local,dest=/tmp/.buildx-cache-new"
        "${build_args[@]:-}"
        "--load"
      )
    fi

    docker "${buildx:-}" build \
           -t "builtins-$target" \
           ${build_args[@]:-} \
           "ci/docker/$target"
    docker run \
           --rm \
           -e RUST_COMPILER_RT_ROOT \
           -e RUSTFLAGS \
           -e "CARGO_TARGET_DIR=/builtins-target" \
           -v "$(pwd):/checkout:ro" \
           -w /checkout \
           ${run_args[@]:-} \
           --init \
           "builtins-$target" \
           sh -c "$run_cmd"
}

if [ "${1:-}" = "--help" ] || [ "$#" -gt 1 ]; then
  set +x
  echo "\
    usage: ./ci/run-docker.sh [target]

    you can also set DOCKER_BASE_IMAGE to use something other than the default
    ubuntu:18.04 (or rustlang/rust:nightly).
  "
  exit
fi

if [ -z "${1:-}" ]; then
  for d in ci/docker/*; do
    run $(basename "$d")
  done
else
  run "$1"
fi
