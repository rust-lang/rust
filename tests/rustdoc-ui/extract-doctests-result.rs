// Test to ensure that it generates expected output for `--output-format=doctest` command-line
// flag.

//@ compile-flags:-Z unstable-options --output-format=doctest
//@ normalize-stdout: "tests/rustdoc-ui" -> "$$DIR"
//@ check-pass

//! ```
//! let x = 12;
//! Ok(())
//! ```
