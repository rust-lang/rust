# The compiler testing framework

The Rust project runs a wide variety of different tests, orchestrated
by the build system (`x.py test`).  The main test harness for testing
the compiler itself is a tool called compiletest (sources in the
[`src/tools/compiletest`]). This section gives a brief overview of how
the testing framework is setup, and then gets into some of the details
on [how to run tests](./running.html#ui) as well as
[how to add new tests](./adding.html).

[`src/tools/compiletest`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest

## Compiletest test suites

The compiletest tests are located in the tree in the [`src/test`]
directory. Immediately within you will see a series of subdirectories
(e.g. `ui`, `run-make`, and so forth). Each of those directories is
called a **test suite** – they house a group of tests that are run in
a distinct mode.

[`src/test`]: https://github.com/rust-lang/rust/tree/master/src/test

Here is a brief summary of the test suites as of this writing and what
they mean. In some cases, the test suites are linked to parts of the manual
that give more details.

- [`ui`](./adding.html#ui) – tests that check the exact
  stdout/stderr from compilation and/or running the test
- `run-pass` – tests that are expected to compile and execute
  successfully (no panics)
  - `run-pass-valgrind` – tests that ought to run with valgrind
- `run-fail` – tests that are expected to compile but then panic
  during execution
- `compile-fail` – tests that are expected to fail compilation.
- `parse-fail` – tests that are expected to fail to parse
- `pretty` – tests targeting the Rust "pretty printer", which
  generates valid Rust code from the AST
- `debuginfo` – tests that run in gdb or lldb and query the debug info
- `codegen` – tests that compile and then test the generated LLVM
  code to make sure that the optimizations we want are taking effect.
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
- `*-fulldeps` – same as above, but indicates that the test depends
  on things other than `libstd` (and hence those things must be built)

## Other Tests

The Rust build system handles running tests for various other things,
including:

- **Tidy** – This is a custom tool used for validating source code
  style and formatting conventions, such as rejecting long lines.
  There is more information in the
  [section on coding conventions](../conventions.html#formatting).

  Example: `./x.py test src/tools/tidy`

- **Unit tests** – The Rust standard library and many of the Rust packages
  include typical Rust `#[test]` unittests.  Under the hood, `x.py` will run
  `cargo test` on each package to run all the tests.

  Example: `./x.py test src/libstd`

- **Doc tests** – Example code embedded within Rust documentation is executed
  via `rustdoc --test`.  Examples:

  `./x.py test src/doc` – Runs `rustdoc --test` for all documentation in
  `src/doc`.

  `./x.py test --doc src/libstd` – Runs `rustdoc --test` on the standard
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

When a Pull Request is opened on Github, [Travis] will automatically launch a
build that will run all tests on a single configuration (x86-64 linux). In
essence, it runs `./x.py test` after building.

The integration bot [bors] is used for coordinating merges to the master
branch. When a PR is approved, it goes into a [queue] where merges are tested
one at a time on a wide set of platforms using Travis and [Appveyor]
(currently over 50 different configurations).  Most platforms only run the
build steps, some run a restricted set of tests, only a subset run the full
suite of tests (see Rust's [platform tiers]).

[Travis]: https://travis-ci.org/rust-lang/rust
[bors]: https://github.com/servo/homu
[queue]: https://buildbot2.rust-lang.org/homu/queue/rust
[Appveyor]: https://ci.appveyor.com/project/rust-lang/rust
[platform tiers]: https://forge.rust-lang.org/platform-support.html

## Testing with Docker images

The Rust tree includes [Docker] image definitions for the platforms used on
Travis in [src/ci/docker].  The script [src/ci/docker/run.sh] is used to build
the Docker image, run it, build Rust within the image, and run the tests.

> TODO: What is a typical workflow for testing/debugging on a platform that
> you don't have easy access to?  Do people build Docker images and enter them
> to test things out?

[Docker]: https://www.docker.com/
[src/ci/docker]: https://github.com/rust-lang/rust/tree/master/src/ci/docker
[src/ci/docker/run.sh]: https://github.com/rust-lang/rust/blob/master/src/ci/docker/run.sh

## Testing on emulators

Some platforms are tested via an emulator for architectures that aren't
readily available.  There is a set of tools for orchestrating running the
tests within the emulator.  Platforms such as `arm-android` and
`arm-unknown-linux-gnueabihf` are set up to automatically run the tests under
emulation on Travis.  The following will take a look at how a target's tests
are run under emulation.

The Docker image for [armhf-gnu] includes [QEMU] to emulate the ARM CPU
architecture.  Included in the Rust tree are the tools [remote-test-client]
and [remote-test-server] which are programs for sending test programs and
libraries to the emulator, and running the tests within the emulator, and
reading the results.  The Docker image is set up to launch
`remote-test-server` and the build tools use `remote-test-client` to
communicate with the server to coordinate running tests (see
[src/bootstrap/test.rs]).

> TODO: What are the steps for manually running tests within an emulator?
> `./src/ci/docker/run.sh armhf-gnu` will do everything, but takes hours to
> run and doesn't offer much help with interacting within the emulator.
>
> Is there any support for emulating other (non-Android) platforms, such as
> running on an iOS emulator?
>
> Is there anything else interesting that can be said here about running tests
> remotely on real hardware?
>
> It's also unclear to me how the wasm or asm.js tests are run.

[armhf-gnu]: https://github.com/rust-lang/rust/tree/master/src/ci/docker/armhf-gnu
[QEMU]: https://www.qemu.org/
[remote-test-client]: https://github.com/rust-lang/rust/tree/master/src/tools/remote-test-client
[remote-test-server]: https://github.com/rust-lang/rust/tree/master/src/tools/remote-test-server
[src/bootstrap/test.rs]: https://github.com/rust-lang/rust/tree/master/src/bootstrap/test.rs

## Crater

[Crater](https://github.com/rust-lang-nursery/crater) is a tool for compiling
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
