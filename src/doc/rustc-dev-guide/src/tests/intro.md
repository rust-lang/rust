# The compiler testing framework

<!-- toc -->

The Rust project runs a wide variety of different tests, orchestrated by
the build system (`x.py test`).  The main test harness for testing the
compiler itself is a tool called compiletest (located in the
[`src/tools/compiletest`] directory). This section gives a brief
overview of how the testing framework is setup, and then gets into some
of the details on [how to run tests](./running.html) as well as [how to
add new tests](./adding.html).

[`src/tools/compiletest`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest

## Compiletest test suites

The compiletest tests are located in the tree in the [`src/test`]
directory. Immediately within you will see a series of subdirectories
(e.g. `ui`, `run-make`, and so forth). Each of those directories is
called a **test suite** – they house a group of tests that are run in
a distinct mode.

[`src/test`]: https://github.com/rust-lang/rust/tree/master/src/test

Here is a brief summary of the test suites and what they mean. In some
cases, the test suites are linked to parts of the manual that give more
details.

- [`ui`](./adding.html#ui) – tests that check the exact
  stdout/stderr from compilation and/or running the test
- `run-pass-valgrind` – tests that ought to run with valgrind
- `pretty` – tests targeting the Rust "pretty printer", which
  generates valid Rust code from the AST
- `debuginfo` – tests that run in gdb or lldb and query the debug info
- `codegen` – tests that compile and then test the generated LLVM
  code to make sure that the optimizations we want are taking effect.
  See [LLVM docs](https://llvm.org/docs/CommandGuide/FileCheck.html) for how to
  write such tests.
- `codegen-units` – tests for the [monomorphization](../backend/monomorph.md)
  collector and CGU partitioning
- `assembly` – similar to `codegen` tests, but verifies assembly output
  to make sure LLVM target backend can handle provided code.
- `mir-opt` – tests that check parts of the generated MIR to make
  sure we are building things correctly or doing the optimizations we
  expect.
- `incremental` – tests for incremental compilation, checking that
  when certain modifications are performed, we are able to reuse the
  results from previous compilations.
- `run-make` – tests that basically just execute a `Makefile`; the
  ultimate in flexibility but quite annoying to write.
- `rustdoc` – tests for rustdoc, making sure that the generated files
  contain the expected documentation.
- `rustfix` – tests for applying [diagnostic
  suggestions](../diagnostics.md#suggestions) with the
  [`rustfix`](https://github.com/rust-lang/rustfix/) crate
- `*-fulldeps` – same as above, but indicates that the test depends
  on things other than `std` (and hence those things must be built)

## Other Tests

The Rust build system handles running tests for various other things,
including:

- **Tidy** – This is a custom tool used for validating source code
  style and formatting conventions, such as rejecting long lines.
  There is more information in the
  [section on coding conventions](../conventions.html#formatting).

  Example: `./x.py test tidy`

- **Formatting** – Rustfmt is integrated with the build system to enforce
  uniform style across the compiler. In the CI, we check that the formatting
  is correct. The formatting check is also automatically run by the Tidy tool
  mentioned above.

  Example: `./x.py fmt --check` checks formatting and exits with an error if
  formatting is needed.

  Example: `./x.py fmt` runs rustfmt on the codebase.

  Example: `./x.py test tidy --bless` does formatting before doing
  other tidy checks.

- **Unit tests** – The Rust standard library and many of the Rust packages
  include typical Rust `#[test]` unittests.  Under the hood, `x.py` will run
  `cargo test` on each package to run all the tests.

  Example: `./x.py test library/std`

- **Doc tests** – Example code embedded within Rust documentation is executed
  via `rustdoc --test`.  Examples:

  `./x.py test src/doc` – Runs `rustdoc --test` for all documentation in
  `src/doc`.

  `./x.py test --doc library/std` – Runs `rustdoc --test` on the standard
  library.

- **Link checker** – A small tool for verifying `href` links within
  documentation.

  Example: `./x.py test src/tools/linkchecker`

- **Dist check** – This verifies that the source distribution tarball created
  by the build system will unpack, build, and run all tests.

  Example: `./x.py test distcheck`

- **Tool tests** – Packages that are included with Rust have all of their
  tests run as well (typically by running `cargo test` within their
  directory).  This includes things such as cargo, clippy, rustfmt, rls, miri,
  bootstrap (testing the Rust build system itself), etc.

- **Cargo test** – This is a small tool which runs `cargo test` on a few
  significant projects (such as `servo`, `ripgrep`, `tokei`, etc.) just to
  ensure there aren't any significant regressions.

  Example: `./x.py test src/tools/cargotest`

## Testing infrastructure

When a Pull Request is opened on Github, [GitHub Actions] will automatically
launch a build that will run all tests on some configurations
(x86_64-gnu-llvm-8 linux. x86_64-gnu-tools linux, mingw-check linux). In
essence, it runs `./x.py test` after building for each of them.

The integration bot [bors] is used for coordinating merges to the master branch.
When a PR is approved, it goes into a [queue] where merges are tested one at a
time on a wide set of platforms using GitHub Actions (as of <!-- date: 2021-01
--> January 2021, over 50 different configurations). Due to the limit on the
number of parallel jobs, we run CI under the [rust-lang-ci] organization except
for PRs. Most platforms only run the build steps, some run a restricted set of
tests, only a subset run the full suite of tests (see Rust's [platform tiers]).

[GitHub Actions]: https://github.com/rust-lang/rust/actions
[rust-lang-ci]: https://github.com/rust-lang-ci/rust/actions
[bors]: https://github.com/servo/homu
[queue]: https://bors.rust-lang.org/queue/rust
[platform tiers]: https://forge.rust-lang.org/release/platform-support.html#rust-platform-support

## Testing with Docker images

The Rust tree includes [Docker] image definitions for the platforms used on
GitHub Actions in [`src/ci/docker`]. The script [`src/ci/docker/run.sh`] is used to build
the Docker image, run it, build Rust within the image, and run the tests.

You can run these images on your local development machine. This can be
helpful to test environments different from your local system. First you will
need to install Docker on a Linux, Windows, or macOS system (typically Linux
will be much faster than Windows or macOS because the later use virtual
machines to emulate a Linux environment). To enter interactive mode which will
start a bash shell in the container, run `src/ci/docker/run.sh --dev <IMAGE>`
where `<IMAGE>` is one of the directory names in `src/ci/docker` (for example
`x86_64-gnu` is a fairly standard Ubuntu environment).

The docker script will mount your local rust source tree in read-only mode,
and an `obj` directory in read-write mode. All of the compiler artifacts will
be stored in the `obj` directory. The shell will start out in the `obj`
directory. From there, you can run `../src/ci/run.sh` which will run the build
as defined by the image.

Alternatively, you can run individual commands to do specific tasks. For
example, you can run `python3 ../x.py test src/test/ui` to just run UI tests.
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

## Running tests on a remote machine

Tests may be run on a remote machine (e.g. to test builds for a different
architecture). This is done using `remote-test-client` on the build machine
to send test programs to `remote-test-server` running on the remote machine.
`remote-test-server` executes the test programs and sends the results back to
the build machine. `remote-test-server` provides *unauthenticated remote code
execution* so be careful where it is used.

To do this, first build `remote-test-server` for the remote
machine, e.g. for RISC-V
```sh
./x.py build src/tools/remote-test-server --target riscv64gc-unknown-linux-gnu
```

The binary will be created at
`./build/$HOST_ARCH/stage2-tools/$TARGET_ARCH/release/remote-test-server`. Copy
this over to the remote machine.

On the remote machine, run the `remote-test-server` with the `remote` argument
(and optionally `-v` for verbose output). Output should look like this:
```sh
$ ./remote-test-server -v remote
starting test server
listening on 0.0.0.0:12345!
```

You can test if the `remote-test-server` is working by connecting to it and
sending `ping\n`. It should reply `pong`:
```sh
$ nc $REMOTE_IP 12345
ping
pong
```

To run tests using the remote runner, set the `TEST_DEVICE_ADDR` environment
variable then use `x.py` as usual. For example, to run `ui` tests for a RISC-V
machine with the IP address `1.2.3.4` use
```sh
export TEST_DEVICE_ADDR="1.2.3.4:12345"
./x.py test src/test/ui --target riscv64gc-unknown-linux-gnu
```

If `remote-test-server` was run with the verbose flag, output on the test machine
may look something like
```
[...]
run "/tmp/work/test1007/a"
run "/tmp/work/test1008/a"
run "/tmp/work/test1009/a"
run "/tmp/work/test1010/a"
run "/tmp/work/test1011/a"
run "/tmp/work/test1012/a"
run "/tmp/work/test1013/a"
run "/tmp/work/test1014/a"
run "/tmp/work/test1015/a"
run "/tmp/work/test1016/a"
run "/tmp/work/test1017/a"
run "/tmp/work/test1018/a"
[...]
```

Tests are built on the machine running `x.py` not on the remote machine. Tests
which fail to build unexpectedly (or `ui` tests producing incorrect build
output) may fail without ever running on the remote machine.

## Testing on emulators

Some platforms are tested via an emulator for architectures that aren't
readily available. For architectures where the standard library is well
supported and the host operating system supports TCP/IP networking, see the
above instructions for testing on a remote machine (in this case the
remote machine is emulated).

There is also a set of tools for orchestrating running the
tests within the emulator. Platforms such as `arm-android` and
`arm-unknown-linux-gnueabihf` are set up to automatically run the tests under
emulation on GitHub Actions. The following will take a look at how a target's tests
are run under emulation.

The Docker image for [armhf-gnu] includes [QEMU] to emulate the ARM CPU
architecture. Included in the Rust tree are the tools [remote-test-client]
and [remote-test-server] which are programs for sending test programs and
libraries to the emulator, and running the tests within the emulator, and
reading the results.  The Docker image is set up to launch
`remote-test-server` and the build tools use `remote-test-client` to
communicate with the server to coordinate running tests (see
[src/bootstrap/test.rs]).

> TODO:
> Is there any support for using an iOS emulator?
>
> It's also unclear to me how the wasm or asm.js tests are run.

[armhf-gnu]: https://github.com/rust-lang/rust/tree/master/src/ci/docker/host-x86_64/armhf-gnu/Dockerfile
[QEMU]: https://www.qemu.org/
[remote-test-client]: https://github.com/rust-lang/rust/tree/master/src/tools/remote-test-client
[remote-test-server]: https://github.com/rust-lang/rust/tree/master/src/tools/remote-test-server
[src/bootstrap/test.rs]: https://github.com/rust-lang/rust/tree/master/src/bootstrap/test.rs

## Crater

[Crater](https://github.com/rust-lang/crater) is a tool for compiling
and running tests for _every_ crate on [crates.io](https://crates.io) (and a
few on GitHub). It is mainly used for checking for extent of breakage when
implementing potentially breaking changes and ensuring lack of breakage by
running beta vs stable compiler versions.

### When to run Crater

You should request a crater run if your PR makes large changes to the compiler
or could cause breakage. If you are unsure, feel free to ask your PR's reviewer.

### Requesting Crater Runs

The rust team maintains a few machines that can be used for running crater runs
on the changes introduced by a PR. If your PR needs a crater run, leave a
comment for the triage team in the PR thread. Please inform the team whether
you require a "check-only" crater run, a "build only" crater run, or a
"build-and-test" crater run. The difference is primarily in time; the
conservative (if you're not sure) option is to go for the build-and-test run.
If making changes that will only have an effect at compile-time (e.g.,
implementing a new trait) then you only need a check run.

Your PR will be enqueued by the triage team and the results will be posted when
they are ready. Check runs will take around ~3-4 days, with the other two
taking 5-6 days on average.

While crater is really useful, it is also important to be aware of a few
caveats:

- Not all code is on crates.io! There is a lot of code in repos on GitHub and
  elsewhere. Also, companies may not wish to publish their code. Thus, a
  successful crater run is not a magically green light that there will be no
  breakage; you still need to be careful.

- Crater only runs Linux builds on x86_64. Thus, other architectures and
  platforms are not tested. Critically, this includes Windows.

- Many crates are not tested. This could be for a lot of reasons, including
  that the crate doesn't compile any more (e.g. used old nightly features),
  has broken or flaky tests, requires network access, or other reasons.

- Before crater can be run, `@bors try` needs to succeed in building artifacts.
  This means that if your code doesn't compile, you cannot run crater.

## Perf runs

A lot of work is put into improving the performance of the compiler and
preventing performance regressions. A "perf run" is used to compare the
performance of the compiler in different configurations for a large collection
of popular crates. Different configurations include "fresh builds", builds
with incremental compilation, etc.

The result of a perf run is a comparison between two versions of the
compiler (by their commit hashes).

You should request a perf run if your PR may affect performance, especially
if it can affect performance adversely.

## Further reading

The following blog posts may also be of interest:

- brson's classic ["How Rust is tested"][howtest]

[howtest]: https://brson.github.io/2017/07/10/how-rust-is-tested
