//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ edition:2018
//@ check-pass

// Prior to setting the default edition for the doctest pre-parser,
// this doctest would fail due to a fatal parsing error.
// see https://github.com/rust-lang/rust/issues/59313

//! ```
//! fn foo() {
//!     drop(async move {});
//! }
//! ```
