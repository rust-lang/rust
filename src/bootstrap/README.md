# Bootstrapping Rust

This is an in-progress README which is targeted at helping to explain how Rust
is bootstrapped and in general some of the technical details of the build
system.

> **Note**: This build system is currently under active development and is not
> intended to be the primarily used one just yet. The makefiles are currently
> the ones that are still "guaranteed to work" as much as possible at least.

## Using the new build system

When configuring Rust via `./configure`, pass the following to enable building
via this build system:

```
./configure --enable-rustbuild
```

## ...

## Directory Layout

This build system houses all output under the `target` directory, which looks
like this:

```
# Root folder of all output. Everything is scoped underneath here
build/

  # Location where the stage0 compiler downloads are all cached. This directory
  # only contains the tarballs themselves as they're extracted elsewhere.
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

  # Each remaining directory is scoped by the "host" triple of compilation at
  # hand.
  x86_64-unknown-linux-gnu/

    # The build artifacts for the `compiler-rt` library for the target this
    # folder is under. The exact layout here will likely depend on the platform,
    # and this is also built with CMake so the build system is also likely
    # different.
    compiler-rt/build/

    # Output folder for LLVM if it is compiled for this target
    llvm/

      # build folder (e.g. the platform-specific build system). Like with
      # compiler-rt this is compiled with CMake
      build/

      # Installation of LLVM. Note that we run the equivalent of 'make install'
      # for LLVM to setup these folders.
      bin/
      lib/
      include/
      share/
      ...

    # Location where the stage0 Cargo and Rust compiler are unpacked. This
    # directory is purely an extracted and overlaid tarball of these two (done
    # by the bootstrapy python script). In theory the build system does not
    # modify anything under this directory afterwards.
    stage0/

    # These to build directories are the cargo output directories for builds of
    # the standard library and compiler, respectively. Internally these may also
    # have other target directories, which represent artifacts being compiled
    # from the host to the specified target.
    #
    # Essentially, each of these directories is filled in by one `cargo`
    # invocation. The build system instruments calling Cargo in the right order
    # with the right variables to ensure these are filled in correctly.
    stageN-std/
    stageN-rustc/

    # This is a special case of the above directories, **not** filled in via
    # Cargo but rather the build system itself. The stage0 compiler already has
    # a set of target libraries for its own host triple (in its own sysroot)
    # inside of stage0/. When we run the stage0 compiler to bootstrap more
    # things, however, we don't want to use any of these libraries (as those are
    # the ones that we're building). So essentially, when the stage1 compiler is
    # being compiled (e.g. after libstd has been built), *this* is used as the
    # sysroot for the stage0 compiler being run.
    #
    # Basically this directory is just a temporary artifact use to configure the
    # stage0 compiler to ensure that the libstd we just built is used to
    # compile the stage1 compiler.
    stage0-rustc/lib/

    # These output directories are intended to be standalone working
    # implementations of the compiler (corresponding to each stage). The build
    # system will link (using hard links) output from stageN-{std,rustc} into
    # each of these directories.
    #
    # In theory there is no extra build output in these directories.
    stage1/
    stage2/
    stage3/
```
