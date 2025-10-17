# GCC codegen backend

If you ran into an error related to tests executed with the GCC codegen backend on CI,
you can use the following command to run tests locally using the GCC backend:

```bash
./x test tests/ui --set 'rust.codegen-backends = ["llvm", "gcc"]' --test-codegen-backend gcc
```

Below, you can find more information about how to configure the GCC backend in bootstrap.

## Choosing which codegen backends are built

The `rust.codegen-backends = [...]` bootstrap option affects which codegen backends will be built and
included in the sysroot of the produced `rustc`. To use the GCC codegen backend, `"gcc"` has to
be included in this array in `bootstrap.toml`:

```toml
rust.codegen-backends = ["llvm", "gcc"]
```

If you don't want to change your `bootstrap.toml` file, you can alternatively run your `x`
commands with `--set 'rust.codegen-backends=["llvm", "gcc"]'`. For example:

```bash
./x build --set 'rust.codegen-backends=["llvm", "gcc"]'
```

The first backend in the `codegen-backends` array will determine which backend will be used as the
*default backend* of the built `rustc`. This also determines which backend will be used to compile the
stage 1 standard library (or anything built in stage 2+). To produce `rustc` that uses the GCC backend
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

Note that in order for this to work, the tested compiler must have the GCC codegen backend available in its sysroot
directory. You can achieve that using the [instructions above](#choosing-which-codegen-backends-are-built).

## Downloading GCC from CI

The `gcc.download-ci-gcc` bootstrap option controls if GCC (which is a dependency of the GCC codegen backend)
will be downloaded from CI or built locally. The default value is `true`, which will download GCC from CI
if there are no local changes to the GCC sources and the given host target is available on CI.

Note that GCC can currently only be downloaded from CI for the `x86_64-unknown-linux-gnu` target.
