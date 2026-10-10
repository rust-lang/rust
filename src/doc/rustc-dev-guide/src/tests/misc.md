# Miscellaneous testing-related info

## `RUSTC_BOOTSTRAP` and stability


This is a bootstrap/compiler implementation detail, but it can also be useful for testing:

- `RUSTC_BOOTSTRAP=1` will "cheat" and bypass usual stability checking, allowing
  you to use unstable features and cli flags on a stable `rustc`.
- `RUSTC_BOOTSTRAP=-1` will force a given `rustc` to pretend it is a stable
  compiler, even if it's actually a nightly `rustc`.
  This is useful because some behaviors of the compiler (e.g. diagnostics)
  can differ depending on whether the compiler is nightly or not.
- `RUSTC_BOOTSTRAP=-2` will force a given `rustc` to pretend it is a stable
  compiler, even if it's actually a nightly `rustc`.
  However, the compiler will still accept unstable compiler flags.
  This is useful for internal testing of the compiler, where we need to pass
  unstable flags required by compiletest, but we also want to test the compiler's
  behavior on the stable channel.

### UI tests

Note that setting `RUSTC_BOOTSTRAP=-1` in ui tests (e.g. via `//@ rustc-env`) is pointless since ui
tests inherit the bootstrap env var from bootstrap (so it's set to `1` by default), and since
compiletest itself passes `-Z` flags (so `-1` breaks the test).

To test stable-only behavior, use the `act-as-stable` directive, which will set `RUSTC_BOOTSTRAP=-2`
for you.

### Run-make tests

For `run-make`/`run-make-cargo` tests, `//@ act-as-stable` is not supported.
You can use the `run_make_support::stable_bare_rustc` function to create an instance of the compiler
that acts as the stable channel (this internally uses `RUSTC_BOOTSTRAP=-1`).
