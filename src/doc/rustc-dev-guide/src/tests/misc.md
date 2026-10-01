# Miscellaneous testing-related info

## `RUSTC_BOOTSTRAP` and stability


This is a bootstrap/compiler implementation detail, but it can also be useful for testing:

- `RUSTC_BOOTSTRAP=1` will "cheat" and bypass usual stability checking, allowing
  you to use unstable features and cli flags on a stable `rustc`.
- `RUSTC_BOOTSTRAP=-1` will force a given `rustc` to pretend it is a stable
  compiler, even if it's actually a nightly `rustc`.
  This is useful because some behaviors of the compiler (e.g. diagnostics)
  can differ depending on whether the compiler is nightly or not.

Note that setting `RUSTC_BOOTSTRAP` in ui tests (e.g. via `//@ rustc-env`) is pointless since ui
tests inherit the bootstrap env var from bootstrap (so it's set to `1` by default), and since
compiletest itself passes `-Z` flags (so `-1` breaks the test). To test stable-only behavior, you
need to write a `run-make` test instead.

For `run-make`/`run-make-cargo` tests, `//@ rustc-env` is not supported.
You can do something like the following for individual `rustc` invocations.

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
