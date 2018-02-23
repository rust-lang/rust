# The compiler testing framework

The Rust project runs a wide variety of different tests, orchestrated by the
build system (`x.py test`).  The main test harness for testing the compiler
itself is a tool called compiletest (sources in the
[`src/tools/compiletest`]). This section gives a brief overview of how the
testing framework is setup, and then gets into some of the details on [how to
run tests](./tests/running.html#ui) as well as [how to add new
tests](./tests/adding.html).

[`src/tools/compiletest`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest

## Compiletest test suites

The compiletest tests are located in the tree in the [`src/test`]
directory. Immediately within you will see a series of subdirectories
(e.g. `ui`, `run-make`, and so forth). Each of those directories is
called a **test suite** -- they house a group of tests that are run in
a distinct mode.

[`src/test`]: https://github.com/rust-lang/rust/tree/master/src/test

Here is a brief summary of the test suites as of this writing and what
they mean. In some cases, the test suites are linked to parts of the manual
that give more details.

- [`ui`](./tests/adding.html#ui) -- tests that check the exact stdout/stderr from compilation
  and/or running the test
- `run-pass` -- tests that are expected to compile and execute successfully (no panics)
  - `run-pass-valgrind` -- tests that ought to run with valrind
- `run-fail` -- tests that are expected to compile but then panic during execution
- `compile-fail` -- tests that are expected to fail compilation.
- `parse-fail` -- tests that are expected to fail to parse
- `pretty` -- tests targeting the Rust "pretty printer", which
  generates valid Rust code from the AST
- `debuginfo` -- tests that run in gdb or lldb and query the debug info
- `codegen` -- tests that compile and then test the generated LLVM
  code to make sure that the optimizations we want are taking effect.
- `mir-opt` -- tests that check parts of the generated MIR to make
  sure we are building things correctly or doing the optimizations we
  expect.
- `incremental` -- tests for incremental compilation, checking that
  when certain modifications are performed, we are able to reuse the
  results from previous compilations.
- `run-make` -- tests that basically just execute a `Makefile`; the
  ultimate in flexibility but quite annoying to write.
- `rustdoc` -- tests for rustdoc, making sure that the generated files contain
  the expected documentation.
- `*-fulldeps` -- same as above, but indicates that the test depends on things other
  than `libstd` (and hence those things must be built)

## Other Tests

The Rust build system handles running tests for various other things,
including:

- **Tidy** -- This is a custom tool used for validating source code style and
  formatting conventions, such as rejecting long lines.  There is more
  information in the [section on coding conventions](./conventions.html#formatting).

  Example: `./x.py test src/tools/tidy`

- **Unittests** -- The Rust standard library and many of the Rust packages
  include typical Rust `#[test]` unittests.  Under the hood, `x.py` will run
  `cargo test` on each package to run all the tests.

  Example: `./x.py test src/libstd`

- **Doctests** -- Example code embedded within Rust documentation is executed
  via `rustdoc --test`.  Examples:

  `./x.py test src/doc` -- Runs `rustdoc --test` for all documentation in
  `src/doc`.

  `./x.py test --doc src/libstd` -- Runs `rustdoc --test` on the standard
  library.

- **Linkchecker** -- A small tool for verifying `href` links within
  documentation.

  Example: `./x.py test src/tools/linkchecker`

- **Distcheck** -- This verifies that the source distribution tarball created
  by the build system will unpack, build, and run all tests.

  Example: `./x.py test distcheck`

- **Tool tests** -- Packages that are included with Rust have all of their
  tests run as well (typically by running `cargo test` within their
  directory).  This includes things such as cargo, clippy, rustfmt, rls, miri,
  bootstrap (testing the Rust build system itself), etc.

- **Cargotest** -- This is a small tool which runs `cargo test` on a few
  significant projects (such as `servo`, `ripgrep`, `tokei`, etc.) just to
  ensure there aren't any significant regressions.

  Example: `./x.py test src/tools/cargotest`

## Testing infrastructure

TODO - bors, platforms, etc.

## Crater

TODO

## Further reading

The following blog posts may also be of interest:

- brson's classic ["How Rust is tested"][howtest]

[howtest]: https://brson.github.io/2017/07/10/how-rust-is-tested
