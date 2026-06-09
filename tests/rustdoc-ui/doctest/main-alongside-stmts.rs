// This test ensures that if there is are any statements alongside a `main` function,
// it will not consider the `main` function as the program entry point but instead
// will generate its own `main` function to wrap everything as it needs to reside in a
// module where only *items* are permitted syntactically.
//
// See <./main-alongside-macro-calls.rs> for comparison.
//
// This is a regression test for:
// * <https://github.com/rust-lang/rust/issues/140162>
// * <https://github.com/rust-lang/rust/issues/139651>
//
//@ compile-flags:--test --test-args --test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

//~v WARN the `main` function of this doctest won't be run
//! ```
//! # if cfg!(miri) { return; }
//! use std::ops::Deref;
//!
//! fn main() {
//!     assert!(false);
//! }
//! ```
//~v WARN the `main` function of this doctest won't be run
//!
//! ```
//! let x = 2;
//! assert_eq!(x, 2);
//!
//! fn main() {
//!     assert!(false);
//! }
//! ```
