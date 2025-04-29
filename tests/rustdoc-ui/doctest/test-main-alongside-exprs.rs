// This test ensures that if there is an expression alongside a `main`
// function, it will not consider the entire code to be part of the `main`
// function and will generate its own function to wrap everything.
//
// This is a regression test for:
// * <https://github.com/rust-lang/rust/issues/140162>
// * <https://github.com/rust-lang/rust/issues/139651>
//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

#![crate_name = "foo"]

//! ```
//! # if cfg!(miri) { return; }
//! use std::ops::Deref;
//!
//! fn main() {
//!     println!("Hi!");
//! }
//! ```
