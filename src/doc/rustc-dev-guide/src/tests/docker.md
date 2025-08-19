# Testing with Docker

The [`src/ci/docker`] directory includes [Docker] image definitions for Linux-based jobs executed on GitHub Actions (non-Linux jobs run outside Docker). You can run these jobs on your local development machine, which can be
helpful to test environments different from your local system. You will
need to install Docker on a Linux, Windows, or macOS system (typically Linux
will be much faster than Windows or macOS because the latter use virtual
machines to emulate a Linux environment).

Jobs running in CI are configured through a set of bash scripts, and it is not always trivial to reproduce their behavior locally. If you want to run a CI job locally in the simplest way possible, you can use a provided helper `citool` that tries to replicate what happens on CI as closely as possible:

```bash
cargo run --manifest-path src/ci/citool/Cargo.toml run-local <job-name>
# For example:
cargo run --manifest-path src/ci/citool/Cargo.toml run-local dist-x86_64-linux-alt
```

If the above script does not work for you, you would like to have more control of the Docker image execution, or you want to understand what exactly happens during Docker job execution, then continue reading below.

## The `run.sh` script
The [`src/ci/docker/run.sh`] script is used to build a specific Docker image, run it,
build Rust within the image, and either run tests or prepare a set of archives designed for distribution. The script will mount your local Rust source tree in read-only mode, and an `obj` directory in read-write mode. All the compiler artifacts will be stored in the `obj` directory. The shell will start out in the `obj`directory. From there, it will execute `../src/ci/run.sh` which starts the build as defined by the Docker image.

You can run `src/ci/docker/run.sh <image-name>` directly. A few important notes regarding the `run.sh` script:
- When executed on CI, the script expects that all submodules are checked out. If some submodule that is accessed by the job is not available, the build will result in an error. You should thus make sure that you have all required submodules checked out locally. You can either do that manually through git, or set `submodules = true` in your `bootstrap.toml` and run a command such as `x build` to let bootstrap download the most important submodules (this might not be enough for the given CI job that you are trying to execute though).
- `<image-name>` corresponds to a single directory located in one of the `src/ci/docker/host-*` directories. Note that image name does not necessarily correspond to a job name, as some jobs execute the same image, but with different environment variables or Docker build arguments (this is a part of the complexity that makes it difficult to run CI jobs locally).
- If you are executing a "dist" job (job beginning with `dist-`), you should set the `DEPLOY=1` environment variable.
- If you are executing an "alternative dist" job (job beginning with `dist-` and ending with `-alt`), you should set the `DEPLOY_ALT=1` environment variable.
- Some of the std tests require IPv6 support. Docker on Linux seems to have it
  disabled by default. Run the commands in [`enable-docker-ipv6.sh`] to enable
  IPv6 before creating the container. This only needs to be done once.

### Interactive mode

Sometimes, it can be useful to build a specific Docker image, and then run custom commands inside it, so that you can experiment with how the given system behaves. You can do that using an interactive mode, which will
start a bash shell in the container, using `src/ci/docker/run.sh --dev <image-name>`.

When inside the Docker container, you can run individual commands to do specific tasks. For
example, you can run `../x test tests/ui` to just run UI tests.

Some additional notes about using the interactive mode:

- The container will be deleted automatically when you exit the shell, however
  the build artifacts persist in the `obj` directory. If you are switching
  between different Docker images, the artifacts from previous environments
  stored in the `obj` directory may confuse the build system. Sometimes you
  will need to delete parts or all of the `obj` directory before building
  inside the container.
- The container is bare-bones, with only a minimal set of packages. You may
  want to install some things like `apt install less vim`.
- You can open multiple shells in the container. First you need the container
  name (a short hash), which is displayed in the shell prompt, or you can run
  `docker container ls` outside of the container to list the available
  containers. With the container name, run `docker exec -it <CONTAINER>
  /bin/bash` where `<CONTAINER>` is the container name like `4ba195e95cef`.

[Docker]: https://www.docker.com/
[`src/ci/docker`]: https://github.com/rust-lang/rust/tree/master/src/ci/docker
[`src/ci/docker/run.sh`]: https://github.com/rust-lang/rust/blob/master/src/ci/docker/run.sh
[`src/ci/run.sh`]: https://github.com/rust-lang/rust/blob/master/src/ci/run.sh
[`enable-docker-ipv6.sh`]: https://github.com/rust-lang/rust/blob/master/src/ci/scripts/enable-docker-ipv6.sh
