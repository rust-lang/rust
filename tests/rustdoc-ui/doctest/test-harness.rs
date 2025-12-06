// Ensure that the code block attr `test_harness` works as expected.
//@ compile-flags: --test --test-args --test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ rustc-env: RUST_BACKTRACE=0
//@ failure-status: 101

// The `main` fn won't be run under `test_harness`, so this test should pass.
//! ```test_harness
//! fn main() {
//!     assert!(false);
//! }
//! ```

// Check that we run both `bad` and `good` even if `bad` fails.
//! ```test_harness
//! #[test]
//! fn bad() {
//!     assert!(false);
//! }
//!
//! #[test]
//! fn good() {
//!     assert!(true);
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
