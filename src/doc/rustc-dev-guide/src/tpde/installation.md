# Installation

TPDE is not currently distributed through `rustup`, and needs to be built locally. The TPDE backend is Linux-only.

## Build instructions

Follow the instructions at [How to build and run the compiler](../building/how-to-build-and-run.md) to clone the `rust-lang/rust` repository and set up a basic bootstrap configuration with the `compiler` profile.

You will also need to enable `llvm.tpde = true` in `bootstrap.toml`, and, due to present limitations with the way LLVM is built, disable CI LLVM downloads with `llvm.download-ci-llvm = false`.

Afterwards, linking the toolchain with rustup will allow you to use it through cargo:

```console
rustup toolchain link stage1 build/host/stage1
# in other crate
RUSTFLAGS="-Zllvm-tpde" cargo +stage1 build
```

You can then run the compiler test suite:

```console
./x test --stage 1 tests/ui --compiletest-rustc-args="-Zllvm-tpde" # or -Zllvm-tpde=only
./x test --stage 1 tests/run-make --compiletest-rustc-args="-Zllvm-tpde" # or -Zllvm-tpde=only
```

## Build instruction for TPDE itself

Following the Rust build instructions above will build the TPDE library used by the Rust compiler.
However, if working directly on TPDE, the [TPDE docs](https://docs.tpde.org/) have instructions on how to build TPDE and run its test suite, which is not exercised by bootstrap.
