# High-level overview of the compiler source

Now that we have [seen what the compiler does][orgch],
let's take a look at the structure of the [`rust-lang/rust`] repository,
where the rustc source code lives.

[`rust-lang/rust`]: https://github.com/rust-lang/rust

> You may find it helpful to read the ["Overview of the compiler"][orgch]
> chapter, which introduces how the compiler works, before this one.

[orgch]: ./overview.md

## Workspace structure

The [`rust-lang/rust`] repository consists of a single large cargo workspace
containing the compiler, the standard libraries ([`core`], [`alloc`], [`std`],
[`proc_macro`], [`etc`]), and [`rustdoc`], along with the build system and a
bunch of tools and submodules for building a full Rust distribution.

The repository consists of three main directories:

- [`compiler/`] contains the source code for `rustc`. It consists of many crates
  that together make up the compiler.
  
- [`library/`] contains the standard libraries ([`core`], [`alloc`], [`std`],
  [`proc_macro`], [`test`]), as well as the Rust runtime ([`backtrace`], [`rtstartup`],
  [`lang_start`]).
  
- [`tests/`] contains the compiler tests.
  
- [`src/`] contains the source code for [`rustdoc`], [`clippy`], [`cargo`], the build system,
  language docs, etc.

[`alloc`]: https://github.com/rust-lang/rust/tree/master/library/alloc
[`backtrace`]: https://github.com/rust-lang/backtrace-rs/
[`cargo`]: https://github.com/rust-lang/cargo
[`clippy`]: https://github.com/rust-lang/rust/tree/master/src/tools/clippy
[`compiler/`]: https://github.com/rust-lang/rust/tree/master/compiler
[`core`]: https://github.com/rust-lang/rust/tree/master/library/core
[`etc`]: https://github.com/rust-lang/rust/tree/master/src/etc
[`lang_start`]: https://github.com/rust-lang/rust/blob/master/library/std/src/rt.rs
[`library/`]: https://github.com/rust-lang/rust/tree/master/library
[`proc_macro`]: https://github.com/rust-lang/rust/tree/master/library/proc_macro
[`rtstartup`]: https://github.com/rust-lang/rust/tree/master/library/rtstartup
[`rust-lang/rust`]: https://github.com/rust-lang/rust
[`rustdoc`]: https://github.com/rust-lang/rust/tree/master/src/tools/rustdoc
[`src/`]: https://github.com/rust-lang/rust/tree/master/src
[`std`]: https://github.com/rust-lang/rust/tree/master/library/std
[`test`]: https://github.com/rust-lang/rust/tree/master/library/test
[`tests/`]: https://github.com/rust-lang/rust/tree/master/tests

## Compiler

The compiler is implemented in the various [`compiler/`] crates.
The [`compiler/`] crates all have names starting with `rustc_*`. These are a
collection of around 50 interdependent crates ranging in size from tiny to
huge. There is also the `rustc` crate which is the actual binary (i.e. the
`main` function); it doesn't actually do anything besides calling the
[`rustc_driver`] crate, which drives the various parts of compilation in other
crates.

The dependency order of these crates is complex, but roughly it is
something like this:

1. `rustc` (the binary) calls [`rustc_driver::main`][main].
1. [`rustc_driver`] depends on a lot of other crates, but the main one is
   [`rustc_interface`].
1. [`rustc_interface`] depends on most of the other compiler crates. It is a
   fairly generic interface for driving the whole compilation.
1. Most of the other `rustc_*` crates depend on [`rustc_middle`], which defines
   a lot of central data structures in the compiler.
1. [`rustc_middle`] and most of the other crates depend on a handful of crates
   representing the early parts of the compiler (e.g. the parser), fundamental
   data structures (e.g. [`Span`]), or error reporting:
   [`rustc_data_structures`], [`rustc_span`], [`rustc_errors`], etc.

[`rustc_data_structures`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_data_structures/index.html
[`rustc_driver`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/index.html
[`rustc_errors`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/index.html
[`rustc_interface`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/index.html
[`rustc_middle`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/index.html
[`rustc_span`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/index.html
[`Span`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/struct.Span.html
[main]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/fn.main.html

You can see the exact dependencies by running `cargo tree`,
just like you would for any other Rust package:

```console
cargo tree --package rustc_driver
```

One final thing: [`src/llvm-project`] is a submodule for our fork of LLVM.
During bootstrapping, LLVM is built and the [`compiler/rustc_llvm`] crate
contains Rust wrappers around LLVM (which is written in C++), so that the
compiler can interface with it.

Most of this book is about the compiler, so we won't have any further
explanation of these crates here.

[`compiler/rustc_llvm`]: https://github.com/rust-lang/rust/tree/master/compiler/rustc_llvm
[`src/llvm-project`]: https://github.com/rust-lang/rust/tree/master/src/
[`Cargo.toml`]: https://github.com/rust-lang/rust/blob/master/Cargo.toml

### Big picture

The dependency structure of the compiler is influenced by two main factors:

1. Organization. The compiler is a _huge_ codebase; it would be an impossibly
   large crate. In part, the dependency structure reflects the code structure
   of the compiler.
2. Compile-time. By breaking the compiler into multiple crates, we can take
   better advantage of incremental/parallel compilation using cargo. In
   particular, we try to have as few dependencies between crates as possible so
   that we don't have to rebuild as many crates if you change one.

At the very bottom of the dependency tree are a handful of crates that are used
by the whole compiler (e.g. [`rustc_span`]). The very early parts of the
compilation process (e.g. [parsing and the Abstract Syntax Tree (`AST`)][parser]) 
depend on only these.

After the [`AST`][parser] is constructed and other early analysis is done, the
compiler's [query system][query] gets set up. The query system is set up in a
clever way using function pointers. This allows us to break dependencies
between crates, allowing more parallel compilation. The query system is defined
in [`rustc_middle`], so nearly all subsequent parts of the compiler depend on
this crate. It is a really large crate, leading to long compile times. Some
efforts have been made to move stuff out of it with varying success. Another
side-effect is that sometimes related functionality gets scattered across
different crates. For example, linting functionality is found across earlier
parts of the crate, [`rustc_lint`], [`rustc_middle`], and other places.

Ideally there would be fewer, more cohesive crates, with incremental and
parallel compilation making sure compile times stay reasonable. However,
incremental and parallel compilation haven't gotten good enough for that yet,
so breaking things into separate crates has been our solution so far.

At the top of the dependency tree is [`rustc_driver`] and [`rustc_interface`]
which is an unstable wrapper around the query system helping drive various
stages of compilation. Other consumers of the compiler may use this interface
in different ways (e.g. [`rustdoc`] or maybe eventually `rust-analyzer`). The
[`rustc_driver`] crate first parses command line arguments and then uses
[`rustc_interface`] to drive the compilation to completion.

[parser]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/index.html
[`rustc_lint`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/index.html
[query]: ./query.md

## rustdoc

The bulk of [`rustdoc`] is in [`librustdoc`]. However, the [`rustdoc`] binary
itself is [`src/tools/rustdoc`], which does nothing except call [`rustdoc::main`].

There is also `JavaScript` and `CSS` for the docs in [`src/tools/rustdoc-js`]
and [`src/tools/rustdoc-themes`]. The type definitions for `--output-format=json`
are in a separate crate in [`src/rustdoc-json-types`].

You can read more about [`rustdoc`] in [this chapter][rustdoc-chapter].

[`librustdoc`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustdoc/index.html
[`rustdoc::main`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustdoc/fn.main.html
[`src/tools/rustdoc-js`]: https://github.com/rust-lang/rust/tree/master/src/tools/rustdoc-js
[`src/tools/rustdoc-themes`]: https://github.com/rust-lang/rust/tree/master/src/tools/rustdoc-themes
[`src/tools/rustdoc`]:  https://github.com/rust-lang/rust/tree/master/src/tools/rustdoc
[`src/rustdoc-json-types`]: https://github.com/rust-lang/rust/tree/master/src/rustdoc-json-types
[rustdoc-chapter]: ./rustdoc.md

## Tests

The test suite for all of the above is in [`tests/`]. You can read more
about the test suite [in this chapter][testsch].

The test harness is in [`src/tools/compiletest/`][`compiletest/`].

[`tests/`]: https://github.com/rust-lang/rust/tree/master/tests
[testsch]: ./tests/intro.md

## Build System

There are a number of tools in the repository just for building the compiler,
standard library, [`rustdoc`], etc, along with testing, building a full Rust
distribution, etc.

One of the primary tools is [`src/bootstrap/`]. You can read more about
bootstrapping [in this chapter][bootstch]. The process may also use other tools
from [`src/tools/`], such as [`tidy/`] or [`compiletest/`].

[`compiletest/`]: https://github.com/rust-lang/rust/tree/master/src/tools/compiletest
[`src/bootstrap/`]: https://github.com/rust-lang/rust/tree/master/src/bootstrap
[`src/tools/`]: https://github.com/rust-lang/rust/tree/master/src/tools
[`tidy/`]: https://github.com/rust-lang/rust/tree/master/src/tools/tidy
[bootstch]: ./building/bootstrapping/intro.md

## Standard library

This code is fairly similar to most other Rust crates except that it must be
built in a special way because it can use unstable ([`nightly`]) features.
The standard library is sometimes referred to as [`libstd or the "standard facade"`].

[`libstd or the "standard facade"`]: https://rust-lang.github.io/rfcs/0040-libstd-facade.html
[`nightly`]: https://doc.rust-lang.org/nightly/nightly-rustc/

## Other

There are a lot of other things in the `rust-lang/rust` repo that are related
to building a full Rust distribution. Most of the time you don't need to worry about them.

These include:
- [`src/ci`]: The CI configuration. This actually quite extensive because we
  run a lot of tests on a lot of platforms.
- [`src/doc`]: Various documentation, including submodules for a few books.
- [`src/etc`]: Miscellaneous utilities.
- And more...

[`src/ci`]: https://github.com/rust-lang/rust/tree/master/src/ci
[`src/doc`]: https://github.com/rust-lang/rust/tree/master/src/doc
[`src/etc`]: https://github.com/rust-lang/rust/tree/master/src/etc
