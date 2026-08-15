//@ compile-flags:--test
//@ edition:2018

// Prior to setting the default edition for the doctest pre-parser,
// this doctest would fail due to a fatal parsing error.
// see https://github.com/rust-lang/rust/issues/59313

//! ```
//! fn foo() {
//!     drop(async move {});
//! }
//! ```
