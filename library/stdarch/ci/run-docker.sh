# Small script to run tests for a target (or all targets) inside all the
# respective docker images.

set -ex

run() {
    echo "Building docker container for TARGET=${1}"
    docker build -t stdsimd ci/docker/$1
    mkdir -p target
    target=$(echo $1 | sed 's/-emulated//')
    echo "Running docker"
    docker run \
      --user `id -u`:`id -g` \
      --rm \
      --init \
      --volume $HOME/.cargo:/cargo \
      --env CARGO_HOME=/cargo \
      --volume `rustc --print sysroot`:/rust:ro \
      --env TARGET=$target \
      --env STDSIMD_TEST_EVERYTHING \
      --volume `pwd`:/checkout:ro \
      --volume `pwd`/target:/checkout/target \
      --workdir /checkout \
      --privileged \
      stdsimd \
      bash \
      -c 'PATH=$PATH:/rust/bin exec ci/run.sh'
}

if [ -z "$1" ]; then
  for d in `ls ci/docker/`; do
    run $d
  done
else
  run $1
fi
