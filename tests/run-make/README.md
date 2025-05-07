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

[`run_make_support`]: ../../src/tools/run-make-support
[extern_prelude]: https://doc.rust-lang.org/reference/names/preludes.html#extern-prelude
