#!/usr/bin/env bash

set -e

export MSYS_NO_PATHCONV=1

script=`cd $(dirname $0) && pwd`/`basename $0`

image=""
dev=0

while [[ $# -gt 0 ]]
do
  case "$1" in
    --dev)
      dev=1
      ;;
    *)
      if [ -n "$image" ]
      then
        echo "expected single argument for the image name"
        exit 1
      fi
      image="$1"
      ;;
  esac
  shift
done

script_dir="`dirname $script`"
docker_dir="${script_dir}/host-$(uname -m)"
ci_dir="`dirname $script_dir`"
src_dir="`dirname $ci_dir`"
root_dir="`dirname $src_dir`"

objdir=$root_dir/obj
dist=$objdir/build/dist

source "$ci_dir/shared.sh"

CACHE_DOMAIN="${CACHE_DOMAIN:-ci-caches.rust-lang.org}"

if [ -f "$docker_dir/$image/Dockerfile" ]; then
    if [ "$CI" != "" ]; then
      hash_key=/tmp/.docker-hash-key.txt
      rm -f "${hash_key}"
      echo $image >> $hash_key

      cat "$docker_dir/$image/Dockerfile" >> $hash_key
      # Look for all source files involves in the COPY command
      copied_files=/tmp/.docker-copied-files.txt
      rm -f "$copied_files"
      for i in $(sed -n -e '/^COPY --from=/! s/^COPY \(.*\) .*$/\1/p' \
          "$docker_dir/$image/Dockerfile"); do
        # List the file names
        find "$script_dir/$i" -type f >> $copied_files
      done
      # Sort the file names and cat the content into the hash key
      sort $copied_files | xargs cat >> $hash_key

      # Include the architecture in the hash key, since our Linux CI does not
      # only run in x86_64 machines.
      uname -m >> $hash_key

      docker --version >> $hash_key
      cksum=$(sha512sum $hash_key | \
        awk '{print $1}')

      url="https://$CACHE_DOMAIN/docker/$cksum"

      echo "Attempting to download $url"
      rm -f /tmp/rustci_docker_cache
      set +e
      retry curl --max-time 600 -y 30 -Y 10 --connect-timeout 30 -f -L -C - \
        -o /tmp/rustci_docker_cache "$url"
      echo "Loading images into docker"
      # docker load sometimes hangs in the CI, so time out after 10 minutes with TERM,
      # KILL after 12 minutes
      loaded_images=$(/usr/bin/timeout -k 720 600 docker load -i /tmp/rustci_docker_cache \
        | sed 's/.* sha/sha/')
      set -e
      echo "Downloaded containers:\n$loaded_images"
    fi

    dockerfile="$docker_dir/$image/Dockerfile"
    if [ -x /usr/bin/cygpath ]; then
        context="`cygpath -w $script_dir`"
        dockerfile="`cygpath -w $dockerfile`"
    else
        context="$script_dir"
    fi
    retry docker \
      build \
      --rm \
      -t rust-ci \
      -f "$dockerfile" \
      "$context"

    if [ "$CI" != "" ]; then
      s3url="s3://$SCCACHE_BUCKET/docker/$cksum"
      upload="aws s3 cp - $s3url"
      digest=$(docker inspect rust-ci --format '{{.Id}}')
      echo "Built container $digest"
      if ! grep -q "$digest" <(echo "$loaded_images"); then
        echo "Uploading finished image to $url"
        set +e
        docker history -q rust-ci | \
          grep -v missing | \
          xargs docker save | \
          gzip | \
          $upload
        set -e
      else
        echo "Looks like docker image is the same as before, not uploading"
      fi
      # Record the container image for reuse, e.g. by rustup.rs builds
      info="$dist/image-$image.txt"
      mkdir -p "$dist"
      echo "$url" >"$info"
      echo "$digest" >>"$info"
    fi
elif [ -f "$docker_dir/disabled/$image/Dockerfile" ]; then
    if isCI; then
        echo Cannot run disabled images on CI!
        exit 1
    fi
    # Transform changes the context of disabled Dockerfiles to match the enabled ones
    tar --transform 's#disabled/#./#' -C $script_dir -c . | docker \
      build \
      --rm \
      -t rust-ci \
      -f "host-$(uname -m)/$image/Dockerfile" \
      -
else
    echo Invalid image: $image

    # Check whether the image exists for other architectures
    for arch_dir in "${script_dir}"/host-*; do
        # Avoid checking non-directories and the current host architecture directory
        if ! [[ -d "${arch_dir}" ]]; then
            continue
        fi
        if [[ "${arch_dir}" = "${docker_dir}" ]]; then
            continue
        fi

        arch_name="$(basename "${arch_dir}" | sed 's/^host-//')"
        if [[ -f "${arch_dir}/${image}/Dockerfile" ]]; then
            echo "Note: the image exists for the ${arch_name} host architecture"
        elif [[ -f "${arch_dir}/disabled/${image}/Dockerfile" ]]; then
            echo "Note: the disabled image exists for the ${arch_name} host architecture"
        else
            continue
        fi
        echo "Note: the current host architecture is $(uname -m)"
    done

    exit 1
fi

mkdir -p $HOME/.cargo
mkdir -p $objdir/tmp
mkdir -p $objdir/cores
mkdir -p /tmp/toolstate

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

# Things get a little weird if this script is already running in a docker
# container. If we're already in a docker container then we assume it's set up
# to do docker-in-docker where we have access to a working `docker` command.
#
# If this is the case (we check via the presence of `/.dockerenv`)
# then we can't actually use the `--volume` argument. Typically we use
# `--volume` to efficiently share the build and source directory between this
# script and the container we're about to spawn. If we're inside docker already
# though the `--volume` argument maps the *host's* folder to the container we're
# about to spawn, when in fact we want the folder in this container itself. To
# work around this we use a recipe cribbed from
# https://circleci.com/docs/2.0/building-docker-images/#mounting-folders to
# create a temporary container with a volume. We then copy the entire source
# directory into this container, and then use that copy in the container we're
# about to spawn. Finally after the build finishes we re-extract the object
# directory.
#
# Note that none of this is necessary if we're *not* in a docker-in-docker
# scenario. If this script is run on a bare metal host then we share a bunch of
# data directories to share as much data as possible. Note that we also use
# `LOCAL_USER_ID` (recognized in `src/ci/run.sh`) to ensure that files are all
# read/written as the same user as the bare-metal user.
if [ -f /.dockerenv ]; then
  docker create -v /checkout --name checkout alpine:3.4 /bin/true
  docker cp . checkout:/checkout
  args="$args --volumes-from checkout"
else
  args="$args --volume $root_dir:/checkout:ro"
  args="$args --volume $objdir:/checkout/obj"
  args="$args --volume $HOME/.cargo:/cargo"
  args="$args --volume $HOME/rustsrc:$HOME/rustsrc"
  args="$args --volume /tmp/toolstate:/tmp/toolstate"

  id=$(id -u)
  if [[ "$id" != 0 && "$(docker -v)" =~ ^podman ]]; then
    # Rootless podman creates a separate user namespace, where an inner
    # LOCAL_USER_ID will map to a different subuid range on the host.
    # The "keep-id" mode maps the current UID directly into the container.
    args="$args --env NO_CHANGE_USER=1 --userns=keep-id"
  else
    args="$args --env LOCAL_USER_ID=$id"
  fi
fi

if [ "$dev" = "1" ]
then
  # Interactive + TTY
  args="$args -it"
  command="/bin/bash"
else
  command="/checkout/src/ci/run.sh"
fi

if [ "$CI" != "" ]; then
  # Get some needed information for $BASE_COMMIT
  #
  # This command gets the last merge commit which we'll use as base to list
  # deleted files since then.
  BASE_COMMIT="$(git log --author=bors@rust-lang.org -n 2 --pretty=format:%H | tail -n 1)"
else
  BASE_COMMIT=""
fi

docker \
  run \
  --workdir /checkout/obj \
  --env SRC=/checkout \
  $args \
  --env CARGO_HOME=/cargo \
  --env DEPLOY \
  --env DEPLOY_ALT \
  --env CI \
  --env TF_BUILD \
  --env BUILD_SOURCEBRANCHNAME \
  --env GITHUB_ACTIONS \
  --env GITHUB_REF \
  --env TOOLSTATE_REPO_ACCESS_TOKEN \
  --env TOOLSTATE_REPO \
  --env TOOLSTATE_PUBLISH \
  --env RUST_CI_OVERRIDE_RELEASE_CHANNEL \
  --env CI_JOB_NAME="${CI_JOB_NAME-$IMAGE}" \
  --env BASE_COMMIT="$BASE_COMMIT" \
  --init \
  --rm \
  rust-ci \
  $command

if [ -f /.dockerenv ]; then
  rm -rf $objdir
  docker cp checkout:/checkout/obj $objdir
fi
