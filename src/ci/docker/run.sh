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

# MacOS reports "arm64" while Linux reports "aarch64". Commonize this.
machine="$(uname -m | sed 's/arm64/aarch64/')"

script_dir="`dirname $script`"
docker_dir="${script_dir}/host-${machine}"
ci_dir="`dirname $script_dir`"
src_dir="`dirname $ci_dir`"
root_dir="`dirname $src_dir`"

source "$ci_dir/shared.sh"

if isCI; then
    objdir=$root_dir/obj
else
    objdir=$root_dir/obj/$image
fi
dist=$objdir/build/dist


if [ -d "$root_dir/.git" ]; then
    IS_GIT_SOURCE=1
fi

CACHE_DOMAIN="${CACHE_DOMAIN:-ci-caches.rust-lang.org}"

if [ -f "$docker_dir/$image/Dockerfile" ]; then
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
    echo "$machine" >> $hash_key

    # Include cache version. Can be used to manually bust the Docker cache.
    echo "2" >> $hash_key

    echo "::group::Image checksum input"
    cat $hash_key
    echo "::endgroup::"

    cksum=$(sha512sum $hash_key | \
    awk '{print $1}')
    echo "Image input checksum ${cksum}"

    dockerfile="$docker_dir/$image/Dockerfile"
    if [ -x /usr/bin/cygpath ]; then
        context="`cygpath -w $script_dir`"
        dockerfile="`cygpath -w $dockerfile`"
    else
        context="$script_dir"
    fi
    echo "::group::Building docker image for $image"

    # Print docker version
    docker --version

    REGISTRY=ghcr.io
    # Default to `rust-lang` to allow reusing the cache for local builds
    REGISTRY_USERNAME=${GITHUB_REPOSITORY_OWNER:-rust-lang}
    # Tag used to push the final Docker image, so that it can be pulled by e.g. rustup
    IMAGE_TAG=${REGISTRY}/${REGISTRY_USERNAME}/rust-ci:${cksum}
    # Tag used to cache the Docker build
    # It seems that it cannot be the same as $IMAGE_TAG, otherwise it overwrites the cache
    CACHE_IMAGE_TAG=${REGISTRY}/${REGISTRY_USERNAME}/rust-ci-cache:${cksum}

    # Docker build arguments.
    build_args=(
        "build"
        "--rm"
        "-t" "rust-ci"
        "-f" "$dockerfile"
        "$context"
    )

    GHCR_BUILDKIT_IMAGE="ghcr.io/rust-lang/buildkit:buildx-stable-1"
    # On non-CI jobs, we try to download a pre-built image from the rust-lang-ci
    # ghcr.io registry. If it is not possible, we fall back to building the image
    # locally.
    if ! isCI;
    then
        if docker pull "${IMAGE_TAG}"; then
            echo "Downloaded Docker image from CI"
            docker tag "${IMAGE_TAG}" rust-ci
        else
            echo "Building local Docker image"
            retry docker "${build_args[@]}"
        fi
    # On PR CI jobs, we don't have permissions to write to the registry cache,
    # but we can still read from it.
    elif [[ "$PR_CI_JOB" == "1" ]];
    then
        # Enable a new Docker driver so that --cache-from works with a registry backend
        # Use a custom image to avoid DockerHub rate limits
        docker buildx create --use --driver docker-container \
          --driver-opt image=${GHCR_BUILDKIT_IMAGE}

        # Build the image using registry caching backend
        retry docker \
          buildx \
          "${build_args[@]}" \
          --cache-from type=registry,ref=${CACHE_IMAGE_TAG} \
          --output=type=docker
    # On auto/try builds, we can also write to the cache.
    else
        # Log into the Docker registry, so that we can read/write cache and the final image
        echo ${DOCKER_TOKEN} | docker login ${REGISTRY} \
            --username ${REGISTRY_USERNAME} \
            --password-stdin

        # Enable a new Docker driver so that --cache-from/to works with a registry backend
        # Use a custom image to avoid DockerHub rate limits
        docker buildx create --use --driver docker-container \
          --driver-opt image=${GHCR_BUILDKIT_IMAGE}

        # Build the image using registry caching backend
        retry docker \
          buildx \
          "${build_args[@]}" \
          --cache-from type=registry,ref=${CACHE_IMAGE_TAG} \
          --cache-to type=registry,ref=${CACHE_IMAGE_TAG},compression=zstd \
          --output=type=docker

        # Print images for debugging purposes
        docker images

        # Tag the built image and push it to the registry
        docker tag rust-ci "${IMAGE_TAG}"
        docker push "${IMAGE_TAG}"

        # Record the container registry tag/url for reuse, e.g. by rustup.rs builds
        # It should be possible to run `docker pull <$IMAGE_TAG>` to download the image
        info="$dist/image-$image.txt"
        mkdir -p "$dist"
        echo "${IMAGE_TAG}" > "$info"
        cat "$info"

        echo "To download the image, run docker pull ${IMAGE_TAG}"
    fi
    echo "::endgroup::"
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
      -f "host-${machine}/$image/Dockerfile" \
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
        echo "Note: the current host architecture is $machine"
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
    args="$args --env AWS_REGION"

    # Disable S3 authentication for PR builds, because the access keys are missing
    if [ "$PR_CI_JOB" != "" ]; then
      args="$args --env SCCACHE_S3_NO_CREDENTIALS=1"
    else
      args="$args --env AWS_ACCESS_KEY_ID"
      args="$args --env AWS_SECRET_ACCESS_KEY"
    fi
else
    mkdir -p $HOME/.cache/sccache
    args="$args --env SCCACHE_DIR=/sccache --volume $HOME/.cache/sccache:/sccache"
fi

# By default, container volumes are bound as read-only; therefore doing experimental work
# or debugging within the container environment (such as fetching submodules and
# building them) is not possible. Setting READ_ONLY_SRC to 0 enables this capability by
# binding the volumes in read-write mode.
if [ "$READ_ONLY_SRC" != "0" ]; then
    SRC_MOUNT_OPTION=":ro"
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
  docker create -v /checkout --name checkout ghcr.io/rust-lang/alpine:3.4 /bin/true
  docker cp . checkout:/checkout
  args="$args --volumes-from checkout"
else
  args="$args --volume $root_dir:/checkout$SRC_MOUNT_OPTION"
  args="$args --volume $objdir:/checkout/obj"
  args="$args --volume $HOME/.cargo:/cargo"
  args="$args --volume /tmp/toolstate:/tmp/toolstate"

  id=$(id -u)
  if [[ "$id" != 0 && "$(docker version)" =~ Podman ]]; then
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
  if [ $IS_GIT_SOURCE -eq 1 ]; then
    command=(/bin/bash -c 'git config --global --add safe.directory /checkout;bash')
  else
    command=(/bin/bash)
  fi
else
  command=(/checkout/src/ci/run.sh)
fi

if isCI; then
  # Get some needed information for $BASE_COMMIT
  #
  # This command gets the last merge commit which we'll use as base to list
  # deleted files since then.
  BASE_COMMIT="$(git log --author=bors@rust-lang.org -n 2 --pretty=format:%H | tail -n 1)"
else
  BASE_COMMIT=""
fi

SUMMARY_FILE=github-summary.md
touch $objdir/${SUMMARY_FILE}

extra_env=""
if [ "$ENABLE_GCC_CODEGEN" = "1" ]; then
  extra_env="$extra_env --env ENABLE_GCC_CODEGEN=1"
  # Fix rustc_codegen_gcc lto issues.
  extra_env="$extra_env --env GCC_EXEC_PREFIX=/usr/lib/gcc/"
  echo "Setting extra environment values for docker: $extra_env"
fi

if [ -n "${DOCKER_SCRIPT}" ]; then
  extra_env="$extra_env --env SCRIPT=\"/scripts/${DOCKER_SCRIPT}\""
fi

docker \
  run \
  --workdir /checkout/obj \
  --env SRC=/checkout \
  $extra_env \
  $args \
  --env CARGO_HOME=/cargo \
  --env DEPLOY \
  --env DEPLOY_ALT \
  --env CI \
  --env GITHUB_ACTIONS \
  --env GITHUB_REF \
  --env GITHUB_STEP_SUMMARY="/checkout/obj/${SUMMARY_FILE}" \
  --env GITHUB_WORKFLOW_RUN_ID \
  --env GITHUB_REPOSITORY \
  --env RUST_BACKTRACE \
  --env TOOLSTATE_REPO_ACCESS_TOKEN \
  --env TOOLSTATE_REPO \
  --env TOOLSTATE_PUBLISH \
  --env RUST_CI_OVERRIDE_RELEASE_CHANNEL \
  --env CI_JOB_NAME="${CI_JOB_NAME-$image}" \
  --env CI_JOB_DOC_URL="${CI_JOB_DOC_URL}" \
  --env BASE_COMMIT="$BASE_COMMIT" \
  --env DIST_TRY_BUILD \
  --env PR_CI_JOB \
  --env OBJDIR_ON_HOST="$objdir" \
  --env CODEGEN_BACKENDS \
  --env DISABLE_CI_RUSTC_IF_INCOMPATIBLE="$DISABLE_CI_RUSTC_IF_INCOMPATIBLE" \
  --init \
  --rm \
  rust-ci \
  "${command[@]}"

if isCI; then
    cat $objdir/${SUMMARY_FILE} >> "${GITHUB_STEP_SUMMARY}"
fi

if [ -f /.dockerenv ]; then
  rm -rf $objdir
  docker cp checkout:/checkout/obj $objdir
fi
