# Bootstrapping Rust

This README is aimed at helping to explain how Rust is bootstrapped,
and some of the technical details of the bootstrap build system.

Note that this README only covers internal information, not how to use the tool.
Please check [bootstrapping dev guide][bootstrapping-dev-guide] for further information.

[bootstrapping-dev-guide]: https://rustc-dev-guide.rust-lang.org/building/bootstrapping/intro.html

## Introduction

The build system defers most of the complicated logic of managing invocations
of rustc and rustdoc to Cargo itself. However, moving through various stages
and copying artifacts is still necessary for it to do. Each time bootstrap
is invoked, it will iterate through the list of predefined steps and execute
each serially in turn if it matches the paths passed or is a default rule.
For each step, bootstrap relies on the step internally being incremental and
parallel. Note, though, that the `-j` parameter to bootstrap gets forwarded
to appropriate test harnesses and such.

## Build phases

Bootstrap build system goes through a few phases to actually build the
compiler. What actually happens when you invoke bootstrap is:

1. The entry point script (`x` for unix like systems, `x.ps1` for windows systems,
   `x.py` cross-platform) is run. This script is responsible for downloading the stage0
   compiler/Cargo binaries, and it then compiles the build system itself (this folder).
   Finally, it then invokes the actual `bootstrap` binary build system.
2. In Rust, `bootstrap` will slurp up all configuration, perform a number of
   sanity checks (whether compilers exist, for example), and then start building the
   stage0 artifacts.
3. The stage0 `cargo`, downloaded earlier, is used to build the standard library
   and the compiler, and then these binaries are then copied to the `stage1`
   directory. That compiler is then used to generate the stage1 artifacts which
   are then copied to the stage2 directory, and then finally, the stage2
   artifacts are generated using that compiler.

The goal of each stage is to (a) leverage Cargo as much as possible and failing
that (b) leverage Rust as much as possible!

## Directory Layout

This build system houses all output under the `build` directory, which looks
like this:

```sh
# Root folder of all output. Everything is scoped underneath here
build/

  # Location where the stage0 compiler downloads are all cached. This directory
  # only contains the tarballs themselves, as they're extracted elsewhere.
  cache/
    2015-12-19/
    2016-01-15/
    2016-01-21/
    ...

  # Output directory for building this build system itself. The stage0
  # cargo/rustc are used to build the build system into this location.
  bootstrap/
    debug/
    release/

  # Output of the dist-related steps like dist-std, dist-rustc, and dist-docs
  dist/

  # Temporary directory used for various input/output as part of various stages
  tmp/

  # Each remaining directory is scoped by the "host" triple of compilation at
  # hand.
  x86_64-unknown-linux-gnu/

    # The build artifacts for the `compiler-rt` library for the target that
    # this folder is under. The exact layout here will likely depend on the
    # platform, and this is also built with CMake, so the build system is
    # also likely different.
    compiler-rt/
      build/

    # Output folder for LLVM if it is compiled for this target
    llvm/

      # build folder (e.g. the platform-specific build system). Like with
      # compiler-rt, this is compiled with CMake
      build/

      # Installation of LLVM. Note that we run the equivalent of 'make install'
      # for LLVM, to setup these folders.
      bin/
      lib/
      include/
      share/
      ...

    # Output folder for all documentation of this target. This is what's filled
    # in whenever the `doc` step is run.
    doc/

    # Output for all compiletest-based test suites
    test/
      ui/
      debuginfo/
      ...

    # Location where the stage0 Cargo and Rust compiler are unpacked. This
    # directory is purely an extracted and overlaid tarball of these two (done
    # by the bootstrap Python script). In theory, the build system does not
    # modify anything under this directory afterwards.
    stage0/

    # These to-build directories are the cargo output directories for builds of
    # the standard library, the test system, the compiler, and various tools,
    # respectively. Internally, these may also
    # have other target directories, which represent artifacts being compiled
    # from the host to the specified target.
    #
    # Essentially, each of these directories is filled in by one `cargo`
    # invocation. The build system instruments calling Cargo in the right order
    # with the right variables to ensure that these are filled in correctly.
    stageN-std/
    stageN-test/
    stageN-rustc/
    stageN-tools/

    # This is a special case of the above directories, **not** filled in via
    # Cargo but rather the build system itself. The stage0 compiler already has
    # a set of target libraries for its own host triple (in its own sysroot)
    # inside of stage0/. When we run the stage0 compiler to bootstrap more
    # things, however, we don't want to use any of these libraries (as those are
    # the ones that we're building). So essentially, when the stage1 compiler is
    # being compiled (e.g. after libstd has been built), *this* is used as the
    # sysroot for the stage0 compiler being run.
    #
    # Basically, this directory is just a temporary artifact used to configure the
    # stage0 compiler to ensure that the libstd that we just built is used to
    # compile the stage1 compiler.
    stage0-sysroot/lib/

    # These output directories are intended to be standalone working
    # implementations of the compiler (corresponding to each stage). The build
    # system will link (using hard links) output from stageN-{std,rustc} into
    # each of these directories.
    #
    # In theory these are working rustc sysroot directories, meaning there is
    # no extra build output in these directories.
    stage1/
    stage2/
    stage3/
```

## Extending bootstrap

When you use bootstrap, you'll call it through the entry point script
(`x`, `x.ps1`, or `x.py`). However, most of the code lives in `src/bootstrap`.
`bootstrap` has a difficult problem: it is written in Rust, but yet it is run
before the Rust compiler is built! To work around this, there are two components
of bootstrap: the main one written in rust, and `bootstrap.py`. `bootstrap.py`
is what gets run by entry point script. It takes care of downloading the `stage0`
compiler, which will then build the bootstrap binary written in Rust.

Because there are two separate codebases behind `x.py`, they need to
be kept in sync. In particular, both `bootstrap.py` and the bootstrap binary
parse `bootstrap.toml` and read the same command line arguments. `bootstrap.py`
keeps these in sync by setting various environment variables, and the
programs sometimes have to add arguments that are explicitly ignored, to be
read by the other.

Some general areas that you may be interested in modifying are:

* Adding a new build tool? Take a look at `bootstrap/src/core/build_steps/tool.rs`
  for examples of other tools.
* Adding a new compiler crate? Look no further! Adding crates can be done by
  adding a new directory with `Cargo.toml`, followed by configuring all
  `Cargo.toml` files accordingly.
* Adding a new dependency from crates.io? This should just work inside the
  compiler artifacts stage (everything other than libtest and libstd).
* Adding a new configuration option? You'll want to modify `bootstrap/src/core/config/flags.rs`
  for command line flags and then `bootstrap/src/core/config/config.rs` to copy the flags to the
  `Config` struct.
* Adding a sanity check? Take a look at `bootstrap/src/core/sanity.rs`.

If you make a major change on bootstrap configuration, please add a new entry to
`CONFIG_CHANGE_HISTORY` in `src/bootstrap/src/utils/change_tracker.rs`.

A 'major change' includes

* A new option or
* A change in the default options.

Changes that do not affect contributors to the compiler or users
building rustc from source don't need an update to `CONFIG_CHANGE_HISTORY`.

If you have any questions, feel free to reach out on the `#t-infra/bootstrap` channel
at [Rust Bootstrap Zulip server][rust-bootstrap-zulip]. When you encounter bugs,
please file issues on the [Rust issue tracker][rust-issue-tracker].

[rust-bootstrap-zulip]: https://rust-lang.zulipchat.com/#narrow/stream/t-infra.2Fbootstrap
[rust-issue-tracker]: https://github.com/rust-lang/rust/issues

## Testing

To run bootstrap tests, execute `x test bootstrap`. If you want to bless snapshot tests, then install `cargo-insta` (`cargo install cargo-insta`) and then run `cargo insta review --manifest-path src/bootstrap/Cargo.toml`.

## Changelog

Because we do not release bootstrap with versions, we also do not maintain CHANGELOG files. To
review the changes made to bootstrap, simply run `git log --no-merges --oneline -- src/bootstrap`.
