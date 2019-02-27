#!/usr/bin/env bash

set -e

export MSYS_NO_PATHCONV=1

script=`cd $(dirname $0) && pwd`/`basename $0`
image=$1

docker_dir="`dirname $script`"
ci_dir="`dirname $docker_dir`"
src_dir="`dirname $ci_dir`"
root_dir="`dirname $src_dir`"

source "$ci_dir/shared.sh"

travis_fold start build_docker
travis_time_start

if [ -f "$docker_dir/$image/Dockerfile" ]; then
    if [ "$CI" != "" ]; then
      hash_key=/tmp/.docker-hash-key.txt
      rm -f "${hash_key}"
      echo $image >> $hash_key
      find $docker_dir -type f | sort | xargs cat >> $hash_key
      docker --version >> $hash_key
      cksum=$(sha512sum $hash_key | \
        awk '{print $1}')
      s3url="s3://$SCCACHE_BUCKET/docker/$cksum"
      url="https://s3-us-west-1.amazonaws.com/$SCCACHE_BUCKET/docker/$cksum"
      echo "Attempting to download $s3url"
      rm -f /tmp/rustci_docker_cache
      set +e
      retry curl -y 30 -Y 10 --connect-timeout 30 -f -L -C - -o /tmp/rustci_docker_cache "$url"
      loaded_images=$(docker load -i /tmp/rustci_docker_cache | sed 's/.* sha/sha/')
      set -e
      echo "Downloaded containers:\n$loaded_images"
    fi

    dockerfile="$docker_dir/$image/Dockerfile"
    if [ -x /usr/bin/cygpath ]; then
        context="`cygpath -w $docker_dir`"
        dockerfile="`cygpath -w $dockerfile`"
    else
        context="$docker_dir"
    fi
    retry docker \
      build \
      --rm \
      -t rust-ci \
      -f "$dockerfile" \
      "$context"

    if [ "$s3url" != "" ]; then
      digest=$(docker inspect rust-ci --format '{{.Id}}')
      echo "Built container $digest"
      if ! grep -q "$digest" <(echo "$loaded_images"); then
        echo "Uploading finished image to $s3url"
        set +e
        docker history -q rust-ci | \
          grep -v missing | \
          xargs docker save | \
          gzip | \
          aws s3 cp - $s3url
        set -e
      else
        echo "Looks like docker image is the same as before, not uploading"
      fi
    fi
elif [ -f "$docker_dir/disabled/$image/Dockerfile" ]; then
    if [ -n "$TRAVIS_OS_NAME" ]; then
        echo Cannot run disabled images on travis!
        exit 1
    fi
    # retry messes with the pipe from tar to docker. Not needed on non-travis
    # Transform changes the context of disabled Dockerfiles to match the enabled ones
    tar --transform 's#^./disabled/#./#' -C $docker_dir -c . | docker \
      build \
      --rm \
      -t rust-ci \
      -f "$image/Dockerfile" \
      -
else
    echo Invalid image: $image
    exit 1
fi

travis_fold end build_docker
travis_time_finish

objdir=$root_dir/obj

mkdir -p $HOME/.cargo
mkdir -p $objdir/tmp
mkdir -p $objdir/cores

args=
if [ "$SCCACHE_BUCKET" != "" ]; then
    args="$args --env SCCACHE_BUCKET"
    args="$args --env SCCACHE_REGION"
    args="$args --env AWS_ACCESS_KEY_ID"
    args="$args --env AWS_SECRET_ACCESS_KEY"
else
    mkdir -p $HOME/.cache/sccache
    args="$args --env SCCACHE_DIR=/sccache --volume $HOME/.cache/sccache:/sccache"
fi

# Run containers as privileged as it should give them access to some more
# syscalls such as ptrace and whatnot. In the upgrade to LLVM 5.0 it was
# discovered that the leak sanitizer apparently needs these syscalls nowadays so
# we'll need `--privileged` for at least the `x86_64-gnu` builder, so this just
# goes ahead and sets it for all builders.
args="$args --privileged"

exec docker \
  run \
  --volume "$root_dir:/checkout:ro" \
  --volume "$objdir:/checkout/obj" \
  --workdir /checkout/obj \
  --env SRC=/checkout \
  $args \
  --env CARGO_HOME=/cargo \
  --env DEPLOY \
  --env DEPLOY_ALT \
  --env LOCAL_USER_ID=`id -u` \
  --env TRAVIS \
  --env TRAVIS_BRANCH \
  --env TOOLSTATE_REPO_ACCESS_TOKEN \
  --env CI_JOB_NAME="${CI_JOB_NAME-$IMAGE}" \
  --volume "$HOME/.cargo:/cargo" \
  --volume "$HOME/rustsrc:$HOME/rustsrc" \
  --init \
  --rm \
  rust-ci \
  /checkout/src/ci/run.sh
