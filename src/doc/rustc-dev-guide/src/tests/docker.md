# Testing with Docker

The Rust tree includes [Docker] image definitions for the platforms used on
GitHub Actions in [`src/ci/docker`].
The script [`src/ci/docker/run.sh`] is used to build the Docker image, run it,
build Rust within the image, and run the tests.

You can run these images on your local development machine. This can be
helpful to test environments different from your local system. First you will
need to install Docker on a Linux, Windows, or macOS system (typically Linux
will be much faster than Windows or macOS because the latter use virtual
machines to emulate a Linux environment). To enter interactive mode which will
start a bash shell in the container, run `src/ci/docker/run.sh --dev <IMAGE>`
where `<IMAGE>` is one of the directory names in `src/ci/docker` (for example
`x86_64-gnu` is a fairly standard Ubuntu environment).

The docker script will mount your local Rust source tree in read-only mode,
and an `obj` directory in read-write mode. All of the compiler artifacts will
be stored in the `obj` directory. The shell will start out in the `obj`
directory. From there, you can run `../src/ci/run.sh` which will run the build
as defined by the image.

Alternatively, you can run individual commands to do specific tasks. For
example, you can run `../x test tests/ui` to just run UI tests.
Note that there is some configuration in the [`src/ci/run.sh`] script that you
may need to recreate. Particularly, set `submodules = false` in your
`config.toml` so that it doesn't attempt to modify the read-only directory.

Some additional notes about using the Docker images:

- Some of the std tests require IPv6 support. Docker on Linux seems to have it
  disabled by default. Run the commands in [`enable-docker-ipv6.sh`] to enable
  IPv6 before creating the container. This only needs to be done once.
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
