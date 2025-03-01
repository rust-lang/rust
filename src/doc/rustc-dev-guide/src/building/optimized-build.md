# Optimized build of the compiler

<!-- toc -->

There are multiple additional build configuration options and techniques that can be used to compile a
build of `rustc` that is as optimized as possible (for example when building `rustc` for a Linux
distribution). The status of these configuration options for various Rust targets is tracked [here].
This page describes how you can use these approaches when building `rustc` yourself.

[here]: https://github.com/rust-lang/rust/issues/103595

## Link-time optimization

Link-time optimization is a powerful compiler technique that can increase program performance. To
enable (Thin-)LTO when building `rustc`, set the `rust.lto` config option to `"thin"`
in `bootstrap.toml`:

```toml
[rust]
lto = "thin"
```

> Note that LTO for `rustc` is currently supported and tested only for
> the `x86_64-unknown-linux-gnu` target. Other targets *may* work, but no guarantees are provided.
> Notably, LTO-optimized `rustc` currently produces [miscompilations] on Windows.

[miscompilations]: https://github.com/rust-lang/rust/issues/109114

Enabling LTO on Linux has [produced] speed-ups by up to 10%.

[produced]: https://github.com/rust-lang/rust/pull/101403#issuecomment-1288190019

## Memory allocator

Using a different memory allocator for `rustc` can provide significant performance benefits. If you
want to enable the `jemalloc` allocator, you can set the `rust.jemalloc` option to `true`
in `bootstrap.toml`:

```toml
[rust]
jemalloc = true
```

> Note that this option is currently only supported for Linux and macOS targets.

## Codegen units

Reducing the amount of codegen units per `rustc` crate can produce a faster build of the compiler.
You can modify the number of codegen units for `rustc` and `libstd` in `bootstrap.toml` with the
following options:

```toml
[rust]
codegen-units = 1
codegen-units-std = 1
```

## Instruction set

By default, `rustc` is compiled for a generic (and conservative) instruction set architecture
(depending on the selected target), to make it support as many CPUs as possible. If you want to
compile `rustc` for a specific instruction set architecture, you can set the `target_cpu` compiler
option in `RUSTFLAGS`:

```bash
RUSTFLAGS="-C target_cpu=x86-64-v3" ./x build ...
```

If you also want to compile LLVM for a specific instruction set, you can set `llvm` flags
in `bootstrap.toml`:

```toml
[llvm]
cxxflags = "-march=x86-64-v3"
cflags = "-march=x86-64-v3"
```

## Profile-guided optimization

Applying profile-guided optimizations (or more generally, feedback-directed optimizations) can
produce a large increase to `rustc` performance, by up to 15% ([1], [2]). However, these techniques
are not simply enabled by a configuration option, but rather they require a complex build workflow
that compiles `rustc` multiple times and profiles it on selected benchmarks.

There is a tool called `opt-dist` that is used to optimize `rustc` with [PGO] (profile-guided
optimizations) and [BOLT] (a post-link binary optimizer) for builds distributed to end users. You
can examine the tool, which is located in `src/tools/opt-dist`, and build a custom PGO build
workflow based on it, or try to use it directly. Note that the tool is currently quite hardcoded to
the way we use it in Rust's continuous integration workflows, and it might require some custom
changes to make it work in a different environment.

[1]: https://blog.rust-lang.org/inside-rust/2020/11/11/exploring-pgo-for-the-rust-compiler.html#final-numbers-and-a-benchmarking-plot-twist
[2]: https://github.com/rust-lang/rust/pull/96978

[PGO]: https://doc.rust-lang.org/rustc/profile-guided-optimization.html

[BOLT]: https://github.com/llvm/llvm-project/blob/main/bolt/README.md

To use the tool, you will need to provide some external dependencies:

- A Python3 interpreter (for executing `x.py`).
- Compiled LLVM toolchain, with the `llvm-profdata` binary. Optionally, if you want to use BOLT,
  the `llvm-bolt` and
  `merge-fdata` binaries have to be available in the toolchain.

These dependencies are provided to `opt-dist` by an implementation of the [`Environment`] struct.
It specifies directories where will the PGO/BOLT pipeline take place, and also external dependencies
like Python or LLVM.

Here is an example of how can `opt-dist` be used locally (outside of CI):

1. Build the tool with the following command:
    ```bash
    ./x build tools/opt-dist
    ```
2. Run the tool with the `local` mode and provide necessary parameters:
    ```bash
    ./build/host/stage0-tools-bin/opt-dist local \
      --target-triple <target> \ # select target, e.g. "x86_64-unknown-linux-gnu"
      --checkout-dir <path>    \ # path to rust checkout, e.g. "."
      --llvm-dir <path>        \ # path to built LLVM toolchain, e.g. "/foo/bar/llvm/install"
      -- python3 x.py dist       # pass the actual build command
    ```
    You can run `--help` to see further parameters that you can modify.

[`Environment`]: https://github.com/rust-lang/rust/blob/ee451f8faccf3050c76cdcd82543c917b40c7962/src/tools/opt-dist/src/environment.rs#L5

> Note: if you want to run the actual CI pipeline, instead of running `opt-dist` locally,
> you can execute `python3 src/ci/github-actions/ci.py run-local dist-x86_64-linux`.
