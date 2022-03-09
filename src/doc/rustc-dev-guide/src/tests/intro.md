# Testing the compiler

<!-- toc -->

The Rust project runs a wide variety of different tests, orchestrated by
the build system (`./x.py test`).
This section gives a brief overview of the different testing tools.
Subsequent chapters dive into [running tests](running.md) and [adding new tests](adding.md).

## Kinds of tests

There are several kinds of tests to exercise things in the Rust distribution.
Almost all of them are driven by `./x.py test`, with some exceptions noted below.

### Compiletest

The main test harness for testing the compiler itself is a tool called [compiletest].
It supports running different styles of tests, called *test suites*.
The tests are all located in the [`src/test`] directory.
The [Compiletest chapter][compiletest] goes into detail on how to use this tool.

> Example: `./x.py test src/test/ui`

[compiletest]: compiletest.md
[`src/test`]: https://github.com/rust-lang/rust/tree/master/src/test

### Package tests

The standard library and many of the compiler packages include typical Rust `#[test]`
unit tests, integration tests, and documentation tests.
You can pass a path to `x.py` to almost any package in the `library` or `compiler` directory,
and `x.py` will essentially run `cargo test` on that package.

Examples:

| Command | Description |
|---------|-------------|
| `./x.py test library/std` | Runs tests on `std` |
| `./x.py test library/core` | Runs tests on `core` |
| `./x.py test compiler/rustc_data_structures` | Runs tests on `rustc_data_structures` |

The standard library relies very heavily on documentation tests to cover its functionality.
However, unit tests and integration tests can also be used as needed.
Almost all of the compiler packages have doctests disabled.

The standard library and compiler always place all unit tests in a separate `tests` file
(this is enforced in [tidy][tidy-unit-tests]).
This approach ensures that when the test file is changed, the crate does not need to be recompiled.
For example:

```rust,ignore
#[cfg(test)]
mod tests;
```

If it wasn't done this way, and the tests were placed in the same file as the source,
then changing or adding a test would cause the crate you are working on to be recompiled.
If you were working on something like `core`,
then that would require recompiling the entire standard library, and the entirety of `rustc`.

`./x.py test` includes some CLI options for controlling the behavior with these tests:

* `--doc` — Only runs documentation tests in the package.
* `--no-doc` — Run all tests *except* documentation tests.

[tidy-unit-tests]: https://github.com/rust-lang/rust/blob/master/src/tools/tidy/src/unit_tests.rs

### Tidy

Tidy is a custom tool used for validating source code style and formatting conventions,
such as rejecting long lines.
There is more information in the [section on coding conventions](../conventions.md#formatting).

> Example: `./x.py test tidy`

### Formatting

Rustfmt is integrated with the build system to enforce uniform style across the compiler.
The formatting check is automatically run by the Tidy tool mentioned above.

Examples:

| Command | Description |
|---------|-------------|
| `./x.py fmt --check` | Checks formatting and exits with an error if formatting is needed. |
| `./x.py fmt` | Runs rustfmt across the entire codebase. |
| `./x.py test tidy --bless` | First runs rustfmt to format the codebase, then runs tidy checks. |

### Book documentation tests

All of the books that are published have their own tests,
primarily for validating that the Rust code examples pass.
Under the hood, these are essentially using `rustdoc --test` on the markdown files.
The tests can be run by passing a path to a book to `./x.py test`.

> Example: `./x.py test src/doc/book`

### Documentation link checker

Links across all documentation is validated with a link checker tool.

> Example: `./x.py test src/tools/linkchecker`

> Example: `./x.py test linkchecker`

This requires building all of the documentation, which might take a while.

### Dist check

`distcheck` verifies that the source distribution tarball created by the build system
will unpack, build, and run all tests.

> Example: `./x.py test distcheck`

### Tool tests

Packages that are included with Rust have all of their tests run as well.
This includes things such as cargo, clippy, rustfmt, rls, miri, bootstrap
(testing the Rust build system itself), etc.

Most of the tools are located in the [`src/tools`] directory.
To run the tool's tests, just pass its path to `./x.py test`.

> Example: `./x.py test src/tools/cargo`

Usually these tools involve running `cargo test` within the tool's directory.

In CI, some tools are allowed to fail.
Failures send notifications to the corresponding teams, and is tracked on the [toolstate website].
More information can be found in the [toolstate documentation].

[`src/tools`]: https://github.com/rust-lang/rust/tree/master/src/tools/
[toolstate documentation]: https://forge.rust-lang.org/infra/toolstate.html
[toolstate website]: https://rust-lang-nursery.github.io/rust-toolstate/

### Cargo test

`cargotest` is a small tool which runs `cargo test` on a few sample projects
(such as `servo`, `ripgrep`, `tokei`, etc.).
This ensures there aren't any significant regressions.

> Example: `./x.py test src/tools/cargotest`

### Crater

Crater is a tool which runs tests on many thousands of public projects.
This tool has its own separate infrastructure for running.
See the [Crater chapter](crater.md) for more details.

### Performance testing

A separate infrastructure is used for testing and tracking performance of the compiler.
See the [Performance testing chapter](perf.md) for more details.

## Further reading

The following blog posts may also be of interest:

- brson's classic ["How Rust is tested"][howtest]

[howtest]: https://brson.github.io/2017/07/10/how-rust-is-tested
