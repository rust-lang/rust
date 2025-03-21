# Testing the compiler

<!-- toc -->

The Rust project runs a wide variety of different tests, orchestrated by the
build system (`./x test`). This section gives a brief overview of the different
testing tools. Subsequent chapters dive into [running tests](running.md) and
[adding new tests](adding.md).

## Kinds of tests

There are several kinds of tests to exercise things in the Rust distribution.
Almost all of them are driven by `./x test`, with some exceptions noted below.

### Compiletest

The main test harness for testing the compiler itself is a tool called
[compiletest].

[compiletest] supports running different styles of tests, organized into *test
suites*. A *test mode* may provide common presets/behavior for a set of *test
suites*. [compiletest]-supported tests are located in the [`tests`] directory.

The [Compiletest chapter][compiletest] goes into detail on how to use this tool.

> Example: `./x test tests/ui`

[compiletest]: compiletest.md
[`tests`]: https://github.com/rust-lang/rust/tree/master/tests

### Package tests

The standard library and many of the compiler packages include typical Rust
`#[test]` unit tests, integration tests, and documentation tests. You can pass a
path to `./x test` for almost any package in the `library/` or `compiler/`
directory, and `x` will essentially run `cargo test` on that package.

Examples:

| Command                                   | Description                           |
|-------------------------------------------|---------------------------------------|
| `./x test library/std`                    | Runs tests on `std` only              |
| `./x test library/core`                   | Runs tests on `core` only             |
| `./x test compiler/rustc_data_structures` | Runs tests on `rustc_data_structures` |

The standard library relies very heavily on documentation tests to cover its
functionality. However, unit tests and integration tests can also be used as
needed. Almost all of the compiler packages have doctests disabled.

All standard library and compiler unit tests are placed in separate `tests` file
(which is enforced in [tidy][tidy-unit-tests]). This ensures that when the test
file is changed, the crate does not need to be recompiled. For example:

```rust,ignore
#[cfg(test)]
mod tests;
```

If it wasn't done this way, and you were working on something like `core`, that
would require recompiling the entire standard library, and the entirety of
`rustc`.

`./x test` includes some CLI options for controlling the behavior with these
package tests:

* `--doc` — Only runs documentation tests in the package.
* `--no-doc` — Run all tests *except* documentation tests.

[tidy-unit-tests]: https://github.com/rust-lang/rust/blob/master/src/tools/tidy/src/unit_tests.rs

### Tidy

Tidy is a custom tool used for validating source code style and formatting
conventions, such as rejecting long lines. There is more information in the
[section on coding conventions](../conventions.md#formatting).

> Examples: `./x test tidy`


### Formatting

Rustfmt is integrated with the build system to enforce uniform style across the
compiler. The formatting check is automatically run by the Tidy tool mentioned
above.

Examples:

| Command                 | Description                                                        |
|-------------------------|--------------------------------------------------------------------|
| `./x fmt --check`       | Checks formatting and exits with an error if formatting is needed. |
| `./x fmt`               | Runs rustfmt across the entire codebase.                           |
| `./x test tidy --bless` | First runs rustfmt to format the codebase, then runs tidy checks.  |

### Book documentation tests

All of the books that are published have their own tests, primarily for
validating that the Rust code examples pass. Under the hood, these are
essentially using `rustdoc --test` on the markdown files. The tests can be run
by passing a path to a book to `./x test`.

> Example: `./x test src/doc/book`

### Documentation link checker

Links across all documentation is validated with a link checker tool.

> Example: `./x test src/tools/linkchecker`

> Example: `./x test linkchecker`

This requires building all of the documentation, which might take a while.

### Dist check

`distcheck` verifies that the source distribution tarball created by the build
system will unpack, build, and run all tests.

> Example: `./x test distcheck`

### Tool tests

Packages that are included with Rust have all of their tests run as well. This
includes things such as cargo, clippy, rustfmt, miri, bootstrap (testing the
Rust build system itself), etc.

Most of the tools are located in the [`src/tools`] directory. To run the tool's
tests, just pass its path to `./x test`.

> Example: `./x test src/tools/cargo`

Usually these tools involve running `cargo test` within the tool's directory.

If you want to run only a specified set of tests, append `--test-args
FILTER_NAME` to the command.

> Example: `./x test src/tools/miri --test-args padding`

In CI, some tools are allowed to fail. Failures send notifications to the
corresponding teams, and is tracked on the [toolstate website]. More information
can be found in the [toolstate documentation].

[`src/tools`]: https://github.com/rust-lang/rust/tree/master/src/tools/
[toolstate documentation]: https://forge.rust-lang.org/infra/toolstate.html
[toolstate website]: https://rust-lang-nursery.github.io/rust-toolstate/

### Ecosystem testing

Rust tests integration with real-world code to catch regressions and make
informed decisions about the evolution of the language. There are several kinds
of ecosystem tests, including Crater. See the [Ecosystem testing
chapter](ecosystem.md) for more details.

### Performance testing

A separate infrastructure is used for testing and tracking performance of the
compiler. See the [Performance testing chapter](perf.md) for more details.

### Codegen backend testing

See [Codegen backend testing](./codegen-backend-tests/intro.md).

## Miscellaneous information

There are some other useful testing-related info at [Misc info](misc.md).

## Further reading

The following blog posts may also be of interest:

- brson's classic ["How Rust is tested"][howtest]

[howtest]: https://brson.github.io/2017/07/10/how-rust-is-tested
