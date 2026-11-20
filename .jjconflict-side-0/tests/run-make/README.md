# The `run-make` test suite

The `run-make` test suite contains tests which are the most flexible out of all the [rust-lang/rust](https://github.com/rust-lang/rust) test suites. `run-make` tests can basically contain arbitrary code, and are supported by the [`run_make_support`] library.

## Infrastructure

A `run-make` test is a test recipe source file `rmake.rs` accompanied by its parent directory (e.g. `tests/run-make/foo/rmake.rs` is the `foo` `run-make` test).

The implementation for collecting and building the `rmake.rs` recipes are in [`src/tools/compiletest/src/runtest.rs`](../../src/tools/compiletest/src/runtest.rs), in `run_rmake_test`.

The setup for the `rmake.rs` can be summarized as a 3-stage process:

1. First, we build the [`run_make_support`] library in bootstrap as a tool lib.
2. Then, we compile the `rmake.rs` "recipe" linking the support library and its dependencies in, and provide a bunch of env vars. We setup a directory structure within `build/<target>/test/run-make/`

   ```
   <test-name>/
       rmake.exe              # recipe binary
       rmake_out/             # sources from test sources copied over
   ```

   and copy non-`rmake.rs` input support files over to `rmake_out/`. The support library is made available as an [*extern prelude*][extern_prelude].
3. Finally, we run the recipe binary and set `rmake_out/` as the working directory.

## External dependencies

`compiletest` passes tool paths and target-specific flags to `rmake.rs` through
environment variables. Prefer using the helpers in [`run_make_support`] over
reading these variables directly. The helpers keep command construction
consistent across hosts and targets, and avoid relying on whichever tools happen
to be first in `PATH`.

Commonly used helpers include:

- `rustc()` and `rustdoc()` for the compiler and rustdoc under test, from
  `RUSTC` and `RUSTDOC`.
- `cc()` and `cxx()` for the target C and C++ compilers, from `CC`/`CXX` plus
  `CC_DEFAULT_FLAGS`/`CXX_DEFAULT_FLAGS`.
- `gcc()` for tests that specifically need `gcc`; unlike `cc()`, this assumes a
  suitable `gcc` is available in `PATH` and does not add `CC_DEFAULT_FLAGS`.
- `llvm_ar()`, `llvm_nm()`, `llvm_objdump()`, `llvm_readobj()`, and similar
  LLVM tools from `LLVM_BIN_DIR`; `llvm_filecheck()` uses `LLVM_FILECHECK`.
- `python_command()` for the Python interpreter selected by bootstrap, from
  `PYTHON`.
- `clang()` for tests that explicitly require clang-based testing, from `CLANG`.
- `cargo()` for in-tree cargo, from `CARGO`; this is only available in the
  `run-make-cargo` and `build-std` suites, not in plain `run-make`.
- `htmldocck()` for the in-tree `src/etc/htmldocck.py` script, invoked through
  `python_command()`.

Some of these variables are only set when the configured builder can provide the
corresponding tool. Tests that require optional tools or LLVM target components
should declare the matching compiletest directive, such as
`//@ needs-force-clang-based-tests` or `//@ needs-llvm-components: x86`, instead
of failing later in the recipe. If a test needs a tool that is not represented by
`run_make_support::external_deps`, add a small helper there rather than open
coding the lookup in individual tests.

[`run_make_support`]: ../../src/tools/run-make-support
[extern_prelude]: https://doc.rust-lang.org/reference/names/preludes.html#extern-prelude
