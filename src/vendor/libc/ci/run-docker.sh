# Small script to run tests for a target (or all targets) inside all the
# respective docker images.

set -ex

run() {
    echo $1
    docker build -t libc ci/docker/$1
    docker run \
      --rm \
      -v `rustc --print sysroot`:/rust:ro \
      -v `pwd`:/checkout:ro \
      -e CARGO_TARGET_DIR=/tmp/target \
      -w /checkout \
      --privileged \
      -it libc \
      ci/run.sh $1
}

if [ -z "$1" ]; then
  for d in `ls ci/docker/`; do
    run $d
  done
else
  run $1
fi
