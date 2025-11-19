# GCC codegen backend

We run a subset of the compiler test suite with the GCC codegen backend on our CI, to help find changes that could break the integration of this backend with the compiler.

If you encounter any bugs or problems with the GCC codegen backend in general, don't hesitate to open issues on the
[`rustc_codegen_gcc` repository](https://github.com/rust-lang/rustc_codegen_gcc).

Note that the backend currently only supports the `x86_64-unknown-linux-gnu` target.

## Running into GCC backend CI errors

If you ran into an error related to tests executed with the GCC codegen backend on CI in the `x86_64-gnu-gcc` job,
you can use the following command to run UI tests locally using the GCC backend, which reproduces what happens on CI:

```bash
./x test tests/ui \
  --set 'rust.codegen-backends = ["llvm", "gcc"]' \
  --set 'rust.debug-assertions = false' \
  --test-codegen-backend gcc
```

If a different test suite has failed on CI, you will have to modify the `tests/ui` part.

To reproduce the whole CI job locally, you can run `cargo run --manifest-path src/ci/citool/Cargo.toml run-local x86_64-gnu-gcc`.
See [Testing with Docker](../docker.md) for more information.

### What to do in case of a GCC job failure?

If the GCC job test fails and it seems like the failure could be caused by the GCC backend, you can ping the [cg-gcc working group](https://github.com/orgs/rust-lang/teams/wg-gcc-backend) using `@rust-lang/wg-gcc-backend`

If fixing a compiler test that fails with the GCC backend is non-trivial, you can ignore that test when executed with `cg_gcc` using the `//@ ignore-backends: gcc` [compiletest directive](../directives.md).

## Choosing which codegen backends are built

The `rust.codegen-backends = [...]` bootstrap option affects which codegen backends will be built and
included in the sysroot of the produced `rustc`.
To use the GCC codegen backend, `"gcc"` has to be included in this array in `bootstrap.toml`:

```toml
rust.codegen-backends = ["llvm", "gcc"]
```

If you don't want to change your `bootstrap.toml` file, you can alternatively run your `x`
commands with `--set 'rust.codegen-backends=["llvm", "gcc"]'`.
For example:

```bash
./x build --set 'rust.codegen-backends=["llvm", "gcc"]'
```

The first backend in the `codegen-backends` array will determine which backend will be used as the
*default backend* of the built `rustc`.
This also determines which backend will be used to compile the
stage 1 standard library (or anything built in stage 2+).
To produce `rustc` that uses the GCC backend
by default, you can thus put `"gcc"` as the first element of this array:

```bash
./x build --set 'rust.codegen-backends=["gcc"]' library
```

## Choosing the codegen backend used in tests

To run compiler tests with the GCC codegen backend being used to build the test Rust programs, you can use the
`--test-codegen-backend` flag:

```bash
./x test tests/ui --test-codegen-backend gcc
```

Note that in order for this to work, the tested compiler must have the GCC codegen backend [available](#choosing-which-codegen-backends-are-built) in its sysroot directory.

## Downloading GCC from CI

The `gcc.download-ci-gcc` bootstrap option controls if GCC (which is a dependency of the GCC codegen backend)
will be downloaded from CI or built locally.
The default value is `true`, which will download GCC from CI
if there are no local changes to the GCC sources and the given host target is available on CI.

## Running tests of the backend itself

In addition to running the compiler's test suites using the GCC codegen backend, you can also run the test suite of the backend itself.

Now you do that using the following command:

```text
./x test rustc_codegen_gcc
```

The backend needs to be [enabled](#choosing-which-codegen-backends-are-built) for this to work.
