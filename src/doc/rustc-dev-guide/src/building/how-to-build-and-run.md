# How to build and run the compiler

The compiler is built using a tool called `x.py`. You will need to
have Python installed to run it.

For instructions on how to install Python and other prerequisites,
see [the `rust-lang/rust` README][readme].

## Get the source code

The main repository is [`rust-lang/rust`][repo]. This contains the compiler,
the standard library (including `core`, `alloc`, `test`, `proc_macro`, etc),
and a bunch of tools (e.g. `rustdoc`, the bootstrapping infrastructure, etc).

[repo]: https://github.com/rust-lang/rust
[readme]: https://github.com/rust-lang/rust#building-on-a-unix-like-system

The very first step to work on `rustc` is to clone the repository:

```bash
git clone https://github.com/rust-lang/rust.git
cd rust
```

## What is `x.py`?

`x.py` is the build tool for the `rust` repository. It can build docs, run tests, and compile the
compiler and standard library.

This chapter focuses on the basics to be productive, but
if you want to learn more about `x.py`, [read this chapter][bootstrap].

[bootstrap]: ./bootstrapping.md

### Running `x.py` slightly more conveniently

There is a binary that wraps `x.py` called `x` in `src/tools/x`. All it does is
run `x.py`, but it can be installed system-wide and run from any subdirectory
of a checkout. It also looks up the appropriate version of `python` to use.

You can install it with `cargo install --path src/tools/x`.

## Create a `config.toml`

To start, run `./x.py setup`. This will do some initialization and create a
`config.toml` for you with reasonable defaults.

Alternatively, you can write `config.toml` by hand. See `config.example.toml` for all the available
settings and explanations of them. See `src/bootstrap/defaults` for common settings to change.

If you have already built `rustc` and you change settings related to LLVM, then you may have to
execute `rm -rf build` for subsequent configuration changes to take effect. Note that `./x.py
clean` will not cause a rebuild of LLVM.

## Common `x.py` commands

Here are the basic invocations of the `x.py` commands most commonly used when
working on `rustc`, `std`, `rustdoc`, and other tools.

| Command | When to use it |
| --- | --- |
| `./x.py check` | Quick check to see if most things compile; [rust-analyzer can run this automatically for you][rust-analyzer] |
| `./x.py build` | Builds `rustc`, `std`, and `rustdoc` |
| `./x.py test` | Runs all tests |
| `./x.py fmt` | Formats all code |

As written, these commands are reasonable starting points. However, there are
additional options and arguments for each of them that are worth learning for
serious development work. In particular, `./x.py build` and `./x.py test`
provide many ways to compile or test a subset of the code, which can save a lot
of time.

Also, note that `x.py` supports all kinds of path suffixes for `compiler`, `library`,
and `src/tools` directories. So, you can simply run `x.py test tidy` instead of
`x.py test src/tools/tidy`. Or, `x.py build std` instead of `x.py build library/std`.

[rust-analyzer]: ./building/suggested.html#configuring-rust-analyzer-for-rustc

See the chapters on [building](./building/how-to-build-and-run.md),
[testing](./tests/intro.md), and [rustdoc](./rustdoc.md) for more details.

### Building the compiler

Note that building will require a relatively large amount of storage space.
You may want to have upwards of 10 or 15 gigabytes available to build the compiler.

Once you've created a `config.toml`, you are now ready to run
`x.py`. There are a lot of options here, but let's start with what is
probably the best "go to" command for building a local compiler:

```bash
./x.py build library
```

This may *look* like it only builds the standard library, but that is not the case.
What this command does is the following:

- Build `std` using the stage0 compiler
- Build `rustc` using the stage0 compiler
  - This produces the stage1 compiler
- Build `std` using the stage1 compiler

This final product (stage1 compiler + libs built using that compiler)
is what you need to build other Rust programs (unless you use `#![no_std]` or
`#![no_core]`).

You will probably find that building the stage1 `std` is a bottleneck for you,
but fear not, there is a (hacky) workaround...
see [the section on avoiding rebuilds for std][keep-stage].

[keep-stage]: ./suggested.md#faster-builds-with---keep-stage

Sometimes you don't need a full build. When doing some kind of
"type-based refactoring", like renaming a method, or changing the
signature of some function, you can use `./x.py check` instead for a much faster build.

Note that this whole command just gives you a subset of the full `rustc`
build. The **full** `rustc` build (what you get with `./x.py build
--stage 2 compiler/rustc`) has quite a few more steps:

- Build `rustc` with the stage1 compiler.
  - The resulting compiler here is called the "stage2" compiler.
- Build `std` with stage2 compiler.
- Build `librustdoc` and a bunch of other things with the stage2 compiler.

You almost never need to do this.

### Build specific components

If you are working on the standard library, you probably don't need to build
the compiler unless you are planning to use a recently added nightly feature.
Instead, you can just build using the bootstrap compiler.

```bash
./x.py build --stage 0 library
```

If you choose the `library` profile when running `x.py setup`, you can omit `--stage 0` (it's the
default).

## Creating a rustup toolchain

Once you have successfully built `rustc`, you will have created a bunch
of files in your `build` directory. In order to actually run the
resulting `rustc`, we recommend creating rustup toolchains. The first
one will run the stage1 compiler (which we built above). The second
will execute the stage2 compiler (which we did not build, but which
you will likely need to build at some point; for example, if you want
to run the entire test suite).

```bash
rustup toolchain link stage0 build/host/stage0-sysroot # beta compiler + stage0 std
rustup toolchain link stage1 build/host/stage1
rustup toolchain link stage2 build/host/stage2
```

Now you can run the `rustc` you built with. If you run with `-vV`, you
should see a version number ending in `-dev`, indicating a build from
your local environment:

```bash
$ rustc +stage1 -vV
rustc 1.48.0-dev
binary: rustc
commit-hash: unknown
commit-date: unknown
host: x86_64-unknown-linux-gnu
release: 1.48.0-dev
LLVM version: 11.0
```

The rustup toolchain points to the specified toolchain compiled in your `build` directory,
so the rustup toolchain will be updated whenever `x.py build` or `x.py test` are run for
that toolchain/stage.

**Note:** the toolchain we've built does not include `cargo`.  In this case, `rustup` will
fall back to using `cargo` from the installed `nightly`, `beta`, or `stable` toolchain
(in that order).  If you need to use unstable `cargo` flags, be sure to run
`rustup install nightly` if you haven't already.  See the
[rustup documentation on custom toolchains](https://rust-lang.github.io/rustup/concepts/toolchains.html#custom-toolchains).

**Note:** rust-analyzer and IntelliJ Rust plugin use a component called
`rust-analyzer-proc-macro-srv` to work with proc macros. If you intend to use a
custom toolchain for a project (e.g. via `rustup override set stage1`) you may
want to build this component:

```bash
./x.py build proc-macro-srv-cli
```

## Building targets for cross-compilation

To produce a compiler that can cross-compile for other targets,
pass any number of `target` flags to `x.py build`.
For example, if your host platform is `x86_64-unknown-linux-gnu`
and your cross-compilation target is `wasm32-wasi`, you can build with:

```bash
./x.py build --target x86_64-unknown-linux-gnu --target wasm32-wasi
```

Note that if you want the resulting compiler to be able to build crates that
involve proc macros or build scripts, you must be sure to explicitly build target support for the
host platform (in this case, `x86_64-unknown-linux-gnu`).

If you want to always build for other targets without needing to pass flags to `x.py build`,
you can configure this in the `[build]` section of your `config.toml` like so:

```toml
[build]
target = ["x86_64-unknown-linux-gnu", "wasm32-wasi"]
```

Note that building for some targets requires having external dependencies installed
(e.g. building musl targets requires a local copy of musl).
Any target-specific configuration (e.g. the path to a local copy of musl)
will need to be provided by your `config.toml`.
Please see `config.example.toml` for information on target-specific configuration keys.

For examples of the complete configuration necessary to build a target, please visit
[the rustc book](https://doc.rust-lang.org/rustc/platform-support.html),
select any target under the "Platform Support" heading on the left,
and see the section related to building a compiler for that target.
For targets without a corresponding page in the rustc book,
it may be useful to [inspect the Dockerfiles](../tests/docker.md)
that the Rust infrastructure itself uses to set up and configure cross-compilation.

If you have followed the directions from the prior section on creating a rustup toolchain,
then once you have built your compiler you will be able to use it to cross-compile like so:

```bash
cargo +stage1 build --target wasm32-wasi
```

## Other `x.py` commands

Here are a few other useful `x.py` commands. We'll cover some of them in detail
in other sections:

- Building things:
  - `./x.py build` – builds everything using the stage 1 compiler,
    not just up to `std`
  - `./x.py build --stage 2` – builds everything with the stage 2 compiler including
    `rustdoc`
- Running tests (see the [section on running tests](../tests/running.html) for
  more details):
  - `./x.py test library/std` – runs the unit tests and integration tests from `std`
  - `./x.py test tests/ui` – runs the `ui` test suite
  - `./x.py test tests/ui/const-generics` - runs all the tests in
  the `const-generics/` subdirectory of the `ui` test suite
  - `./x.py test tests/ui/const-generics/const-types.rs` - runs
  the single test `const-types.rs` from the `ui` test suite

### Cleaning out build directories

Sometimes you need to start fresh, but this is normally not the case.
If you need to run this then `rustbuild` is most likely not acting right and
you should file a bug as to what is going wrong. If you do need to clean
everything up then you only need to run one command!

```bash
./x.py clean
```

`rm -rf build` works too, but then you have to rebuild LLVM, which can take
a long time even on fast computers.
