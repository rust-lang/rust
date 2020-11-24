# Bootstrapping the Compiler

This subchapter is about the bootstrapping process.

## What is bootstrapping? How does it work?

[Bootstrapping] is the process of using a compiler to compile itself.
More accurately, it means using an older compiler to compile a newer version
of the same compiler.

This raises a chicken-and-egg paradox: where did the first compiler come from?
It must have been written in a different language. In Rust's case it was
[written in OCaml][ocaml-compiler]. However it was abandoned long ago and the
only way to build a modern version of rustc is a slightly less modern
version.

This is exactly how `x.py` works: it downloads the current beta release of
rustc, then uses it to compile the new compiler.

## Stages of bootstrapping

Compiling `rustc` is done in stages:

- **Stage 0:** the stage0 compiler is usually (you can configure `x.py` to use
  something else) the current _beta_ `rustc` compiler and its associated dynamic
  libraries (which `x.py` will download for you). This stage0 compiler is then
  used only to compile `rustbuild`, `std`, and `rustc`. When compiling
  `rustc`, this stage0 compiler uses the freshly compiled `std`.
  There are two concepts at play here: a compiler (with its set of dependencies)
  and its 'target' or 'object' libraries (`std` and `rustc`).
  Both are staged, but in a staggered manner.
- **Stage 1:** the code in your clone (for new version) is then
  compiled with the stage0 compiler to produce the stage1 compiler.
  However, it was built with an older compiler (stage0), so to
  optimize the stage1 compiler we go to next the stage.
  - In theory, the stage1 compiler is functionally identical to the
    stage2 compiler, but in practice there are subtle differences. In
    particular, the stage1 compiler itself was built by stage0 and
    hence not by the source in your working directory: this means that
    the symbol names used in the compiler source may not match the
    symbol names that would have been made by the stage1 compiler. This is
    important when using dynamic linking and the lack of ABI compatibility
    between versions. This primarily manifests when tests try to link with any
    of the `rustc_*` crates or use the (now deprecated) plugin infrastructure.
    These tests are marked with `ignore-stage1`.
- **Stage 2:** we rebuild our stage1 compiler with itself to produce
  the stage2 compiler (i.e. it builds itself) to have all the _latest
  optimizations_. (By default, we copy the stage1 libraries for use by
  the stage2 compiler, since they ought to be identical.)
- _(Optional)_ **Stage 3**: to sanity check our new compiler, we
  can build the libraries with the stage2 compiler. The result ought
  to be identical to before, unless something has broken.

The `stage2` compiler is the one distributed with `rustup` and all other
install methods. However, it takes a very long time to build because one must
first build the new compiler with an older compiler and then use that to
build the new compiler with itself. For development, you usually only want
the `stage1` compiler: `x.py build library/std`.

### Default stages

`x.py` tries to be helpful and pick the stage you most likely meant for each subcommand.
These defaults are as follows:

- `doc`: `--stage 0`
- `build`: `--stage 1`
- `test`: `--stage 1`
- `dist`: `--stage 2`
- `install`: `--stage 2`
- `bench`: `--stage 2`

You can always override the stage by passing `--stage N` explicitly.

For more information about stages, [see below](#understanding-stages-of-bootstrap).

## Complications of bootstrapping

Since the build system uses the current beta compiler to build the stage-1
bootstrapping compiler, the compiler source code can't use some features
until they reach beta (because otherwise the beta compiler doesn't support
them). On the other hand, for [compiler intrinsics][intrinsics] and internal
features, the features _have_ to be used. Additionally, the compiler makes
heavy use of nightly features (`#![feature(...)]`). How can we resolve this
problem?

There are two methods used:
1. The build system sets `--cfg bootstrap` when building with `stage0`, so we
can use `cfg(not(bootstrap))` to only use features when built with `stage1`.
This is useful for e.g. features that were just stabilized, which require
`#![feature(...)]` when built with `stage0`, but not for `stage1`.
2. The build system sets `RUSTC_BOOTSTRAP=1`. This special variable means to
_break the stability guarantees_ of rust: Allow using `#![feature(...)]` with
a compiler that's not nightly. This should never be used except when
bootstrapping the compiler.

[Bootstrapping]: https://en.wikipedia.org/wiki/Bootstrapping_(compilers)
[intrinsics]: ../appendix/glossary.md#intrinsic
[ocaml-compiler]: https://github.com/rust-lang/rust/tree/ef75860a0a72f79f97216f8aaa5b388d98da6480/src/boot

## Contributing to bootstrap

When you use the bootstrap system, you'll call it through `x.py`.
However, most of the code lives in `src/bootstrap`.
`bootstrap` has a difficult problem: it is written in Rust, but yet it is run
before the rust compiler is built! To work around this, there are two
components of bootstrap: the main one written in rust, and `bootstrap.py`.
`bootstrap.py` is what gets run by x.py. It takes care of downloading the
`stage0` compiler, which will then build the bootstrap binary written in
Rust.

Because there are two separate codebases behind `x.py`, they need to
be kept in sync. In particular, both `bootstrap.py` and the bootstrap binary
parse `config.toml` and read the same command line arguments. `bootstrap.py`
keeps these in sync by setting various environment variables, and the
programs sometimes have to add arguments that are explicitly ignored, to be
read by the other.

### Adding a setting to config.toml

This section is a work in progress. In the meantime, you can see an example
contribution [here][bootstrap-build].

[bootstrap-build]: https://github.com/rust-lang/rust/pull/71994

## Understanding stages of bootstrap

### Overview

This is a detailed look into the separate bootstrap stages.

The convention `x.py` uses is that:
- A `--stage N` flag means to run the stage N compiler (`stageN/rustc`).
- A "stage N artifact" is a build artifact that is _produced_ by the stage N compiler.
- The "stage (N+1) compiler" is assembled from "stage N artifacts". This
  process is called _uplifting_.

#### Build artifacts

Anything you can build with `x.py` is a _build artifact_.
Build artifacts include, but are not limited to:

- binaries, like `stage0-rustc/rustc-main`
- shared objects, like `stage0-sysroot/rustlib/libstd-6fae108520cf72fe.so`
- [rlib] files, like `stage0-sysroot/rustlib/libstd-6fae108520cf72fe.rlib`
- HTML files generated by rustdoc, like `doc/std`

[rlib]: ../serialization.md

#### Assembling the compiler

There is a separate step between building the compiler and making it possible
to run. This step is called _assembling_ or _uplifting_ the compiler. It copies
all the necessary build artifacts from `build/stageN-sysroot` to
`build/stage(N+1)`, which allows you to use `build/stage(N+1)` as a [toolchain]
with `rustup toolchain link`.

There is [no way to trigger this step on its own][#73519], but `x.py` will
perform it automatically any time you build with stage N+1.

[toolchain]: https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run.html#creating-a-rustup-toolchain
[#73519]: https://github.com/rust-lang/rust/issues/73519

#### Examples

- `x.py build --stage 0` means to build with the beta `rustc`.
- `x.py doc --stage 0` means to document using the beta `rustdoc`.
- `x.py test --stage 0 library/std` means to run tests on the standard library
    without building `rustc` from source ('build with stage 0, then test the
  artifacts'). If you're working on the standard library, this is normally the
  test command you want.
- `x.py test src/test/ui` means to build the stage 1 compiler and run
  `compiletest` on it. If you're working on the compiler, this is normally the
  test command you want.

#### Examples of what *not* to do

- `x.py test --stage 0 src/test/ui` is not meaningful: it runs tests on the
  _beta_ compiler and doesn't build `rustc` from source. Use `test src/test/ui`
  instead, which builds stage 1 from source.
- `x.py test --stage 0 compiler/rustc` builds the compiler but runs no tests:
  it's running `cargo test -p rustc`, but cargo doesn't understand Rust's
  tests. You shouldn't need to use this, use `test` instead (without arguments).
- `x.py build --stage 0 compiler/rustc` builds the compiler, but does
  not [assemble] it. Use `x.py build library/std` instead, which puts the
  compiler in `stage1/rustc`.

[assemble]: #assembling-the-compiler

### Building vs. Running


Note that `build --stage N compiler/rustc` **does not** build the stage N compiler:
instead it builds the stage _N+1_ compiler _using_ the stage N compiler.

In short, _stage 0 uses the stage0 compiler to create stage0 artifacts which
will later be uplifted to be the stage1 compiler_.

In each stage, two major steps are performed:

1. `std` is compiled by the stage N compiler.
2. That `std` is linked to programs built by the stage N compiler, including
   the stage N artifacts (stage (N+1) compiler).

This is somewhat intuitive if one thinks of the stage N artifacts as "just"
another program we are building with the stage N compiler:
`build --stage N compiler/rustc` is linking the stage N artifacts to the `std`
built by the stage N compiler.

Here is a chart of a full build using `x.py`:

<img alt="A diagram of the rustc compilation phases" src="../img/rustc_stages.svg" class="center" />

Keep in mind this diagram is a simplification, i.e. `rustdoc` can be built at
different stages, the process is a bit different when passing flags such as
`--keep-stage`, or if there are non-host targets.

The stage 2 compiler is what is shipped to end-users.

### Stages and `std`

Note that there are two `std` libraries in play here:
1. The library _linked_ to `stageN/rustc`, which was built by stage N-1 (stage N-1 `std`)
2. The library _used to compile programs_ with `stageN/rustc`, which was
   built by stage N (stage N `std`).

Stage N `std` is pretty much necessary for any useful work with the stage N compiler.
Without it, you can only compile programs with `#![no_core]` -- not terribly useful!

The reason these need to be different is because they aren't necessarily ABI-compatible:
there could be a new layout optimizations, changes to MIR, or other changes
to Rust metadata on nightly that aren't present in beta.

This is also where `--keep-stage 1 library/std` comes into play. Since most
changes to the compiler don't actually change the ABI, once you've produced a
`std` in stage 1, you can probably just reuse it with a different compiler.
If the ABI hasn't changed, you're good to go, no need to spend time
recompiling that `std`.
`--keep-stage` simply assumes the previous compile is fine and copies those
artifacts into the appropriate place, skipping the cargo invocation.

### Cross-compiling

Building stage2 `std` is different depending on whether you are cross-compiling or not
(see in the table how stage2 only builds non-host `std` targets).
This is because `x.py` uses a trick: if `HOST` and `TARGET` are the same,
it will reuse stage1 `std` for stage2! This is sound because stage1 `std`
was compiled with the stage1 compiler, i.e. a compiler using the source code
you currently have checked out. So it should be identical (and therefore ABI-compatible)
to the `std` that `stage2/rustc` would compile.

However, when cross-compiling, stage1 `std` will only run on the host.
So the stage2 compiler has to recompile `std` for the target.

### Why does only libstd use `cfg(bootstrap)`?

The `rustc` generated by the stage0 compiler is linked to the freshly-built
`std`, which means that for the most part only `std` needs to be cfg-gated,
so that `rustc` can use features added to std immediately after their addition,
without need for them to get into the downloaded beta.

Note this is different from any other Rust program: stage1 `rustc`
is built by the _beta_ compiler, but using the _master_ version of libstd!

The only time `rustc` uses `cfg(bootstrap)` is when it adds internal lints
that use diagnostic items. This happens very rarely.

### Directories and artifacts generated by x.py

The following tables indicate the outputs of various stage actions:

| Stage 0 Action                                            | Output                                       |
|-----------------------------------------------------------|----------------------------------------------|
| `beta` extracted                                          | `build/HOST/stage0`                          |
| `stage0` builds `bootstrap`                               | `build/bootstrap`                            |
| `stage0` builds `test`/`std`                              | `build/HOST/stage0-std/TARGET`               |
| copy `stage0-std` (HOST only)                             | `build/HOST/stage0-sysroot/lib/rustlib/HOST` |
| `stage0` builds `rustc` with `stage0-sysroot`             | `build/HOST/stage0-rustc/HOST`               |
| copy `stage0-rustc (except executable)`                   | `build/HOST/stage0-sysroot/lib/rustlib/HOST` |
| build `llvm`                                              | `build/HOST/llvm`                            |
| `stage0` builds `codegen` with `stage0-sysroot`           | `build/HOST/stage0-codegen/HOST`             |
| `stage0` builds `rustdoc`, `clippy`, `miri`, with `stage0-sysroot` | `build/HOST/stage0-tools/HOST`      |

`--stage=0` stops here.

| Stage 1 Action                                      | Output                                |
|-----------------------------------------------------|---------------------------------------|
| copy (uplift) `stage0-rustc` executable to `stage1` | `build/HOST/stage1/bin`               |
| copy (uplift) `stage0-codegen` to `stage1`          | `build/HOST/stage1/lib`               |
| copy (uplift) `stage0-sysroot` to `stage1`          | `build/HOST/stage1/lib`               |
| `stage1` builds `test`/`std`                        | `build/HOST/stage1-std/TARGET`        |
| copy `stage1-std` (HOST only)                       | `build/HOST/stage1/lib/rustlib/HOST`  |
| `stage1` builds `rustc`                             | `build/HOST/stage1-rustc/HOST`        |
| copy `stage1-rustc` (except executable)             | `build/HOST/stage1/lib/rustlib/HOST`  |
| `stage1` builds `codegen`                           | `build/HOST/stage1-codegen/HOST`      |

`--stage=1` stops here.

| Stage 2 Action                                         | Output                                                          |
|--------------------------------------------------------|-----------------------------------------------------------------|
| copy (uplift) `stage1-rustc` executable                | `build/HOST/stage2/bin`                                         |
| copy (uplift) `stage1-sysroot`                         | `build/HOST/stage2/lib and build/HOST/stage2/lib/rustlib/HOST`  |
| `stage2` builds `test`/`std` (not HOST targets)        | `build/HOST/stage2-std/TARGET`                                  |
| copy `stage2-std` (not HOST targets)                   | `build/HOST/stage2/lib/rustlib/TARGET`                          |
| `stage2` builds `rustdoc`, `clippy`, `miri`            | `build/HOST/stage2-tools/HOST`                                  |
| copy `rustdoc`                                         | `build/HOST/stage2/bin`                                         |

`--stage=2` stops here.

## Passing stage-specific flags to `rustc`

`x.py` allows you to pass stage-specific flags to `rustc` when bootstrapping.
The `RUSTFLAGS_BOOTSTRAP` environment variable is passed as RUSTFLAGS to the bootstrap stage
(stage0), and `RUSTFLAGS_NOT_BOOTSTRAP` is passed when building artifacts for later stages.

## Environment Variables

During bootstrapping, there are a bunch of compiler-internal environment
variables that are used. If you are trying to run an intermediate version of
`rustc`, sometimes you may need to set some of these environment variables
manually. Otherwise, you get an error like the following:

```text
thread 'main' panicked at 'RUSTC_STAGE was not set: NotPresent', library/core/src/result.rs:1165:5
```

If `./stageN/bin/rustc` gives an error about environment variables, that
usually means something is quite wrong -- or you're trying to compile e.g.
`rustc` or `std` or something that depends on environment variables. In
the unlikely case that you actually need to invoke rustc in such a situation,
you can find the environment variable values by adding the following flag to
your `x.py` command: `--on-fail=print-env`.
