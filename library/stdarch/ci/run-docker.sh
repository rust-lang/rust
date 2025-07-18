#!/usr/bin/env sh

# Small script to run tests for a target (or all targets) inside all the
# respective docker images.

set -ex

if [ $# -lt 1 ]; then
    >&2 echo "Usage: $0 <TARGET>"
    exit 1
fi

run() {
    # Set the linker that is used for the host (e.g. when compiling a build.rs)
    # This overrides any configuration in e.g. `.cargo/config.toml`, which will
    # probably not work within the docker container.
    HOST_LINKER="CARGO_TARGET_$(rustc --print host-tuple | tr '[:lower:]-' '[:upper:]_')_LINKER"

    # Prevent `Read-only file system (os error 30)`.
    cargo generate-lockfile

    echo "Building docker container for TARGET=${1}"
    docker build -t stdarch -f "ci/docker/${1}/Dockerfile" ci/
    mkdir -p target c_programs rust_programs
    echo "Running docker"
    # shellcheck disable=SC2016
    docker run \
      --rm \
      --user "$(id -u)":"$(id -g)" \
      --env CARGO_HOME=/cargo \
      --env CARGO_TARGET_DIR=/checkout/target \
      --env TARGET="${1}" \
      --env "${HOST_LINKER}"="cc" \
      --env STDARCH_TEST_EVERYTHING \
      --env STDARCH_DISABLE_ASSERT_INSTR \
      --env NOSTD \
      --env NORUN \
      --env RUSTFLAGS \
      --env CARGO_UNSTABLE_BUILD_STD \
      --env RUST_STD_DETECT_UNSTABLE \
      --volume "${HOME}/.cargo":/cargo \
      --volume "$(rustc --print sysroot)":/rust:ro \
      --volume "$(pwd)":/checkout:ro \
      --volume "$(pwd)"/target:/checkout/target \
      --volume "$(pwd)"/c_programs:/checkout/c_programs \
      --volume "$(pwd)"/rust_programs:/checkout/rust_programs \
      --init \
      --workdir /checkout \
      --privileged \
      stdarch \
      sh -c "HOME=/tmp PATH=\$PATH:/rust/bin exec ci/run.sh ${1}"
}

if [ -z "$1" ]; then
  for d in ci/docker/*; do
    run "${d}"
  done
else
  run "${1}"
fi
