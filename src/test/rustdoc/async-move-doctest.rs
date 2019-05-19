// compile-flags:--test
// edition:2018

// prior to setting the default edition for the doctest pre-parser, this doctest would fail due to
// a fatal parsing error
// see https://github.com/rust-lang/rust/issues/59313

//! ```
//! #![feature(async_await)]
//!
//! fn foo() {
//!     drop(async move {});
//! }
//! ```
