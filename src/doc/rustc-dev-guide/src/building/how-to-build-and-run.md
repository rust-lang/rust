# How to Build and Run the Compiler

The compiler is built using a tool called `x.py`. You will need to
have Python installed to run it. But before we get to that, if you're going to
be hacking on `rustc`, you'll want to tweak the configuration of the compiler.
The default configuration is oriented towards running the compiler as a user,
not a developer.

For instructions on how to install Python and other prerequisites,
see [the next page](./prerequisites.md).

## Get the source code

The very first step to work on `rustc` is to clone the repository:

```bash
git clone https://github.com/rust-lang/rust.git
cd rust
```

## Create a `config.toml`

To start, copy [`config.toml.example`] to `config.toml`:

[`config.toml.example`]: https://github.com/rust-lang/rust/blob/master/config.toml.example

```bash
cp config.toml.example config.toml
```

Then you will want to open up the file and change the following
settings (and possibly others, such as `llvm.ccache`):

```toml
[llvm]
# Whether to use Rust CI built LLVM instead of locally building it.
download-ci-llvm = true

# Indicates whether the LLVM assertions are enabled or not
assertions = true

[rust]
# Whether or not to leave debug! and trace! calls in the rust binary.
# Overrides the `debug-assertions` option, if defined.
#
# Defaults to rust.debug-assertions value
#
# If you see a message from `tracing` saying
# `max_level_info` is enabled and means logging won't be shown,
# set this value to `true`.
debug-logging = true

# Whether to always use incremental compilation when building rustc
incremental = true
```

If you have already built `rustc`, then you may have to execute `rm -rf build` for subsequent
configuration changes to take effect. Note that `./x.py clean` will not cause a
rebuild of LLVM, so if your configuration change affects LLVM, you will need to
manually `rm -rf build/` before rebuilding.

## What is `x.py`?

`x.py` is the script used to orchestrate the tooling in the `rustc` repository.
It is the script that can build docs, run tests, and compile `rustc`.
It is the now preferred way to build `rustc` and it replaces the old makefiles
from before. Below are the different ways to utilize `x.py` in order to
effectively deal with the repo for various common tasks.

This chapter focuses on the basics to be productive, but
if you want to learn more about `x.py`, read its README.md
[here](https://github.com/rust-lang/rust/blob/master/src/bootstrap/README.md).
To read more about the bootstrap process and why `x.py` is necessary,
[read this chapter][bootstrap].

### Running `x.py` slightly more conveniently

There is a binary that wraps `x.py` called `x` in `src/tools/x`. All it does is
run `x.py`, but it can be installed system-wide and run from any subdirectory
of a checkout. It also looks up the appropriate version of `python` to use.

You can install it with `cargo install --path src/tools/x`.

[bootstrap]: ./bootstrapping.md

## Building the Compiler

To build a compiler, run `./x.py build`. This will build up to the stage1 compiler,
including `rustdoc`, producing a usable compiler toolchain from the source
code you have checked out.

Note that building will require a relatively large amount of storage space.
You may want to have upwards of 10 or 15 gigabytes available to build the compiler.

There are many flags you can pass to the build command of `x.py` that can be
beneficial to cutting down compile times or fitting other things you might
need to change. They are:

```txt
Options:
    -v, --verbose       use verbose output (-vv for very verbose)
    -i, --incremental   use incremental compilation
        --config FILE   TOML configuration file for build
        --build BUILD   build target of the stage0 compiler
        --host HOST     host targets to build
        --target TARGET target targets to build
        --on-fail CMD   command to run on failure
        --stage N       stage to build
        --keep-stage N  stage to keep without recompiling
        --src DIR       path to the root of the rust checkout
    -j, --jobs JOBS     number of jobs to run in parallel
    -h, --help          print this help message
```

For hacking, often building the stage 1 compiler is enough, but for
final testing and release, the stage 2 compiler is used.

`./x.py check` is really fast to build the rust compiler.
It is, in particular, very useful when you're doing some kind of
"type-based refactoring", like renaming a method, or changing the
signature of some function.

Once you've created a `config.toml`, you are now ready to run
`x.py`. There are a lot of options here, but let's start with what is
probably the best "go to" command for building a local rust:

```bash
./x.py build -i library/std
```

This may *look* like it only builds `std`, but that is not the case.
What this command does is the following:

- Build `std` using the stage0 compiler (using incremental)
- Build `rustc` using the stage0 compiler (using incremental)
  - This produces the stage1 compiler
- Build `std` using the stage1 compiler (cannot use incremental)

This final product (stage1 compiler + libs built using that compiler)
is what you need to build other rust programs (unless you use `#![no_std]` or
`#![no_core]`).

The command includes the `-i` switch which enables incremental compilation.
This will be used to speed up the first two steps of the process:
in particular, if you make a small change, we ought to be able to use your old
results to make producing the stage1 **compiler** faster.

Unfortunately, incremental cannot be used to speed up making the
stage1 libraries.  This is because incremental only works when you run
the *same compiler* twice in a row.  In this case, we are building a
*new stage1 compiler* every time. Therefore, the old incremental
results may not apply. **As a result, you will probably find that
building the stage1 `std` is a bottleneck for you** -- but fear not,
there is a (hacky) workaround.  See [the section on "recommended
workflows"](./suggested.md) below.

Note that this whole command just gives you a subset of the full `rustc`
build. The **full** `rustc` build (what you get if you say `./x.py build
--stage 2 compiler/rustc`) has quite a few more steps:

- Build `rustc` with the stage1 compiler.
  - The resulting compiler here is called the "stage2" compiler.
- Build `std` with stage2 compiler.
- Build `librustdoc` and a bunch of other things with the stage2 compiler.

<a name=toolchain></a>

## Build specific components

- Build only the core library

```bash
./x.py build --stage 0 library/core
```

- Build only the core and `proc_macro` libraries

```bash
./x.py build --stage 0 library/core library/proc_macro
```

Sometimes you might just want to test if the part you’re working on can
compile. Using these commands you can test that it compiles before doing
a bigger build to make sure it works with the compiler. As shown before
you can also pass flags at the end such as `--stage`.

## Creating a rustup toolchain

Once you have successfully built `rustc`, you will have created a bunch
of files in your `build` directory. In order to actually run the
resulting `rustc`, we recommend creating rustup toolchains. The first
one will run the stage1 compiler (which we built above). The second
will execute the stage2 compiler (which we did not build, but which
you will likely need to build at some point; for example, if you want
to run the entire test suite).

```bash
rustup toolchain link stage1 build/<host-triple>/stage1
rustup toolchain link stage2 build/<host-triple>/stage2
```

The `<host-triple>` would typically be one of the following:

- Linux: `x86_64-unknown-linux-gnu`
- Mac: `x86_64-apple-darwin`
- Windows: `x86_64-pc-windows-msvc`

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
## Other `x.py` commands

Here are a few other useful `x.py` commands. We'll cover some of them in detail
in other sections:

- Building things:
  - `./x.py build` – builds everything using the stage 1 compiler,
    not just up to `std`
  - `./x.py build --stage 2` – builds the stage2 compiler
- Running tests (see the [section on running tests](../tests/running.html) for
  more details):
  - `./x.py test library/std` – runs the `#[test]` tests from `std`
  - `./x.py test src/test/ui` – runs the `ui` test suite
  - `./x.py test src/test/ui/const-generics` - runs all the tests in
  the `const-generics/` subdirectory of the `ui` test suite
  - `./x.py test src/test/ui/const-generics/const-types.rs` - runs
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
