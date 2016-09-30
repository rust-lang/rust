# Small script to run tests for a target (or all targets) inside all the
# respective docker images.

set -ex

run() {
    local gid=$(id -g) \
          group=$(id -g -n) \
          target=$1 \
          uid=$(id -u) \
          user=$(id -u -n)

    echo $target
    docker build -t $target ci/docker/$target
    docker run \
           --rm \
           -e CARGO_HOME=/cargo \
           -e CARGO_TARGET_DIR=/target \
           -v $HOME/.cargo:/cargo \
           -v `pwd`/target:/target \
           -v `pwd`:/checkout:ro \
           -v `rustc --print sysroot`:/rust:ro \
           -w /checkout \
           -it $target \
           sh -c "
groupadd -g $gid $group
useradd -m -g $gid -u $uid $user
chown $user /cargo /target
su -c 'PATH=\$PATH:/rust/bin ci/run.sh $target' $user
"
}

if [ -z "$1" ]; then
  for d in `ls ci/docker/`; do
    run $d
  done
else
  run $1
fi
