# High-level overview of the compiler source

> **NOTE**: The structure of the repository is going through a lot of
> transitions. In particular, we want to get to a point eventually where the
> top-level directory has separate directories for the compiler, build-system,
> std libs, etc, rather than one huge `src/` directory.

## Workspace structure

The `rust-lang/rust` repository consists of a single large cargo workspace
containing the compiler, the standard library (core, alloc, std, etc), and
`rustdoc`, along with the build system and bunch of tools and submodules for
building a full Rust distribution.

As of this writing, this structure is gradually undergoing some transformation
to make it a bit less monolithic and more approachable, especially to
newcommers.

> Eventually, the hope is for the standard library to live in a `stdlib/`
> directory, while the compiler lives in `compiler/`. However, as of this
> writing, both live in `src/`.

The repository consists of a `src` directory, under which there live many
crates, which are the source for the compiler, standard library, etc, as
mentioned above.

## Standard library

The standard library crates are obviously named `libstd`, `libcore`,
`liballoc`, etc. There is also `libproc_macro`, `libtest`, and other runtime
libraries.

This code is fairly similar to most other Rust crates except that it must be
built in a special way because it can use unstable features.

## Compiler

The compiler crates all have names starting with `librustc_*`. These are a large
collection of interdependent crates. There is also the `rustc` crate which is
the actual binary. It doesn't actually do anything besides calling the compiler
main function elsewhere.

The dependency structure of these crates is complex, but roughly it is
something like this:

- `rustc` (the binary) calls [`rustc_driver::main`][main].
    - [`rustc_driver`] depends on a lot of other crates, but the main one is
      [`rustc_interface`].
        - [`rustc_interface`] depends on most of the other compiler crates. It
          is a fairly generic interface for driving the whole compilation.
            - The most of the other `rustc_*` crates depend on [`rustc_middle`],
              which defines a lot of central data structures in the compiler.
                - [`rustc_middle`] and most of the other crates depend on a
                  handful of crates representing the early parts of the
                  compiler (e.g. the parser), fundamental data structures (e.g.
                  [`Span`]), or error reporting: [`rustc_data_strucutres`],
                  [`rustc_span`], [`rustc_errors`], etc.

[main]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/fn.main.html
[`rustc_driver`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/index.html
[`rustc_interface`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/index.html
[`rustc_middle`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/index.html
[`rustc_data_strucutres`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_data_strucutres/index.html
[`rustc_span`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/index.html
[`Span`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/struct.Span.html
[`rustc_errors`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/index.html

You can see the exact dependencies by reading the `Cargo.toml` for the various
crates, just like a normal Rust crate.

You may ask why the compiler is broken into so many crates. There are two major reasons:

1. Organization. The compiler is a _huge_ codebase; it would be an impossibly large crate.
2. Compile time. By breaking the compiler into multiple crates, we can take
   better advantage of incremental/parallel compilation using cargo. In
   particular, we try to have as few dependencies between crates as possible so
   that we dont' have to rebuild as many crates if you change one.

Most of this book is about the compiler, so we won't have any further
explanation of these crates here.

One final thing: [`src/llvm-project`] is a submodule for our fork of LLVM.

[`src/llvm-project`]: https://github.com/rust-lang/rust/tree/master/src

## rustdoc

The bulk of `rustdoc` is in [`librustdoc`]. However, the `rustdoc` binary
itself is [`src/tools/rustdoc`], which does nothing except call [`rustdoc::main`].

There is also javascript and CSS for the rustdocs in [`src/tools/rustdoc-js`]
and [`src/tools/rustdoc-themes`].

You can read more about rustdoc in [this chapter][rustdocch].

[`librustdoc`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustdoc/index.html
[`rustdoc::main`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustdoc/fn.main.html
[`src/tools/rustdoc`]:  https://github.com/rust-lang/rust/tree/master/src/tools/rustdoc
[`src/tools/rustdoc-js`]: https://github.com/rust-lang/rust/tree/master/src/tools/rustdoc-js
[`src/tools/rustdoc-themes`]: https://github.com/rust-lang/rust/tree/master/src/tools/rustdoc-themes

[rustdocch]: ./rustdoc-internals.md

## Tests

The test suite for all of the above is in [`src/test/`]. You can read more
about the test suite [in this chapter][testsch].

The test harness itself is in [`src/tools/compiletest`].

[testsch]: ./tests/intro.md

[`src/test/`]: https://github.com/rust-lang/rust/tree/master/src/test
[`src/tools/compiletest`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest

## Build System

There are a number of tools in the repository just for building the compiler,
standard library, rustdoc, etc, along with testing, building a full Rust
distribution, etc.

One of the primary tools is [`src/bootstrap`]. You can read more about
bootstrapping [in this chapter][bootstch]. The process may also use other tools
from `src/tools/`, such as [`tidy`] or [`compiletest`].

[`src/bootstrap`]: https://github.com/rust-lang/rust/tree/master/src/bootstrap
[`tidy`]: https://github.com/rust-lang/rust/tree/master/src/tools/tidy
[`compiletest`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest

[bootstch]: ./building/bootstrapping.md

## Other

There are a lot of other things in the `rust-lang/rust` repo that are related
to building a full rust distribution. Most of the time you don't need to worry
about them.

These include:
- [`src/ci`]: The CI configuration. This actually quite extensive because we
  run a lot of tests on a lot of platforms.
- [`src/doc`]: Various documentation, including submodules for a few books.
- [`src/etc`]: Miscellaneous utilities.
- [`src/tools/rustc-workspace-hack`], and others: Various workarounds to make cargo work with bootstrapping.
- And more...

[`src/ci`]: https://github.com/rust-lang/rust/tree/master/src/ci
[`src/doc`]: https://github.com/rust-lang/rust/tree/master/src/doc
[`src/etc`]: https://github.com/rust-lang/rust/tree/master/src/etc
[`src/tools/rustc-workspace-hack`]: https://github.com/rust-lang/rust/tree/master/src/tools/rustc-workspace-hack
