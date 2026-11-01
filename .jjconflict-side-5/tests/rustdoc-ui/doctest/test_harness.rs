// Ensure that the code block attr `test_harness` works as expected.
//@ compile-flags: --test --test-args --test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "Test executable failed \(.+?\)" -> "Test executable failed ($$STATUS)"
//@ rustc-env: RUST_BACKTRACE=0
//@ failure-status: 101

// NOTE(#157511): This test file only contains `test_harness` doctests that each only contain at
// most a single `#[test]` function. That's because at the time of writing, arguments passed via
// `--test-args` only get propagated to the "main" libtest runner. However, we would *have* to pass
// `--test-threads=1` to the inner libtest runner to ensure deterministic test output. We can only
// do that by utilizing a runtool currently.
//
// While we could call a runtool "inline" (via something like `sh -c` on Linux) here in this UI test
// we would need to branch on the host platform and do some crimes to get it working. Instead, we
// have chosen to use a run-make test for this with a Rust runtool which is cross-platform.

// The `main` fn won't be run under `test_harness`, so this test should pass.
//! ```test_harness
//! fn main() {
//!     assert!(false);
//! }
//! ```

// Contrary to normal doctests, cfg `test` is active under `test_harness`.
//! ```test_harness
//! #[cfg(test)]
//! mod group {
//!     #[test]
//!     fn element() {
//!         assert!(false);
//!     }
//! }
//! ```

// `test_harness` doctests aren't implicitly wrapped in a `main` fn even if they contain stmts.
//! ```test_harness
//! assert!(true);
//!
//! #[test] fn extra() {}
//! ```
