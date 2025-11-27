# Miscellaneous testing-related info

## `RUSTC_BOOTSTRAP` and stability

<!-- date-check: Nov 2024 -->

This is a bootstrap/compiler implementation detail, but it can also be useful
for testing:

- `RUSTC_BOOTSTRAP=1` will "cheat" and bypass usual stability checking, allowing
  you to use unstable features and cli flags on a stable `rustc`.
- `RUSTC_BOOTSTRAP=-1` will force a given `rustc` to pretend it is a stable
  compiler, even if it's actually a nightly `rustc`. This is useful because some
  behaviors of the compiler (e.g. diagnostics) can differ depending on whether
  the compiler is nightly or not.

In `ui` tests and other test suites that support `//@ rustc-env`, you can specify

```rust,ignore
// Force unstable features to be usable on stable rustc
//@ rustc-env:RUSTC_BOOTSTRAP=1

// Or force nightly rustc to pretend it is a stable rustc
//@ rustc-env:RUSTC_BOOTSTRAP=-1
```

For `run-make`/`run-make-cargo` tests, `//@ rustc-env` is not supported. You can do
something like the following for individual `rustc` invocations.

```rust,ignore
use run_make_support::rustc;

fn main() {
    rustc()
        // Pretend that I am very stable
        .env("RUSTC_BOOTSTRAP", "-1")
        //...
        .run();
}
```

## `RUSTC_TEST_GIT_HASH` env var

> Context:
>
> - <https://github.com/rust-lang/rust/issues/132875>
> - <https://github.com/rust-lang/rust/issues/132845>

`RUSTC_TEST_GIT_HASH` is a special env var used intentionally to bypass
bootstrap to make sure that the built rustc and rustdoc binaries correctly
report git hash information. We can't rely on information reported by bootstrap
regarding git hash availability because the bootstrap logic itself can be wrong
(see linked issues above).

- This env var must be set in CI.
- To run `tests/run-make/version-verbose-commit-hash` locally, set
  `RUSTC_TEST_GIT_HASH` to any non-empty value. Otherwise, this test will be
  skipped.
