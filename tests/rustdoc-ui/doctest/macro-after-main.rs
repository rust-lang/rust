// This test checks a corner case where the macro calls used to be skipped,
// making them considered as statement, and therefore some cases where
// `include!` macro was then put into a function body, making the doctest
// compilation fail.

//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

//! ```
//! include!("./auxiliary/macro-after-main.rs");
//!
//! fn main() {}
//! eprintln!();
//! ```
