# Small script to run tests for a target (or all targets) inside all the
# respective docker images.

set -ex

run() {
    echo $1
    CMD="cargo test --target $1"
    if [ "$NO_RUN" = "1" ]; then
        CMD="$CMD --no-run"
    fi
    docker build -t libc ci/docker/$1
    docker run \
      -v `rustc --print sysroot`:/rust:ro \
      -v `pwd`:/checkout:ro \
      -e CARGO_TARGET_DIR=/tmp/target \
      -w /checkout \
      --privileged \
      -it libc \
      bash -c "$CMD && $CMD --release"
}

if [ -z "$1" ]; then
  for d in `ls ci/docker/`; do
    run $d
  done
else
  run $1
fi
