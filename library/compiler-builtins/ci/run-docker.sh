#!/bin/bash

# Small script to run tests for a target (or all targets) inside all the
# respective docker images.

set -euxo pipefail

host_arch="$(uname -m | sed 's/arm64/aarch64/')"

# Directories and files that do not yet exist need to be created before
# calling docker, otherwise docker will create them but they will be owned
# by root.
mkdir -p target
cargo generate-lockfile
cargo generate-lockfile --manifest-path builtins-test-intrinsics/Cargo.toml

run() {
    local target="$1"

    echo "testing target: $target"

    emulated=""
    target_arch="$(echo "$target" | cut -d'-' -f1)"
    if [ "$target_arch" != "$host_arch" ]; then
        emulated=1
        echo "target is emulated"
    fi

    run_cmd="HOME=/tmp"

    if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
        # Enable Docker image caching on GHA
        build_cmd=("buildx" "build")
        build_args=(
            "--cache-from" "type=local,src=/tmp/.buildx-cache"
            "--cache-to" "type=local,dest=/tmp/.buildx-cache-new"
            # This is the beautiful bash syntax for expanding an array but neither
            # raising an error nor returning an empty string if the array is empty.
            "${build_args[@]:+"${build_args[@]}"}"
            "--load"
        )
    fi

    if [ "$(uname -s)" = "Linux" ] && [ -z "${DOCKER_BASE_IMAGE:-}" ]; then
        # Share the host rustc and target. Do this only on Linux and if the image
        # isn't overridden
        run_args=(
            --user "$(id -u):$(id -g)"
            -e "CARGO_HOME=/cargo"
            -v "${HOME}/.cargo:/cargo"
            -v "$(pwd)/target:/builtins-target"
            -v "$(rustc --print sysroot):/rust:ro"
        )
        run_cmd="$run_cmd PATH=\$PATH:/rust/bin:/cargo/bin"
    else
        # Use rustc provided by a docker image
        docker volume create compiler-builtins-cache
        build_args=(
            "--build-arg"
            "IMAGE=${DOCKER_BASE_IMAGE:-rustlang/rust:nightly}"
        )
        run_args=(-v "compiler-builtins-cache:/builtins-target")
        run_cmd="$run_cmd HOME=/tmp" "USING_CONTAINER_RUSTC=1"
    fi

    if [ -d compiler-rt ]; then
        export RUST_COMPILER_RT_ROOT="/checkout/compiler-rt"
    fi

    run_cmd="$run_cmd ci/run.sh $target"

    docker "${build_cmd[@]:-build}" \
        -t "builtins-$target" \
        "${build_args[@]:-}" \
        "ci/docker/$target"
    docker run \
        --rm \
        -e CI \
        -e CARGO_TARGET_DIR=/builtins-target \
        -e CARGO_TERM_COLOR \
        -e MAY_SKIP_LIBM_CI \
        -e RUSTFLAGS \
        -e RUST_BACKTRACE \
        -e RUST_COMPILER_RT_ROOT \
        -e "EMULATED=$emulated" \
        -v "$(pwd):/checkout:ro" \
        -w /checkout \
        "${run_args[@]:-}" \
        --init \
        "builtins-$target" \
        sh -c "$run_cmd"
}

if [ "${1:-}" = "--help" ] || [ "$#" -gt 1 ]; then
    set +x
    echo "\
    usage: ./ci/run-docker.sh [target]

    you can also set DOCKER_BASE_IMAGE to use something other than the default
    ubuntu:24.04 (or rustlang/rust:nightly).
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
