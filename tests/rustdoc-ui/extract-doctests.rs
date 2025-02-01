// Test to ensure that it generates expected output for `--output-format=doctest` command-line
// flag.

//@ compile-flags:-Z unstable-options --output-format=doctest
//@ normalize-stdout: "tests/rustdoc-ui" -> "$$DIR"
//@ check-pass

//! ```ignore (checking attributes)
//! let x = 12;
//! let y = 14;
//! ```
//!
//! ```edition2018,compile_fail
//! let
//! ```
