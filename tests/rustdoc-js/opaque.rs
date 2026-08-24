// This test ensures that rustdoc doesn't crash when building the search index
// when an opaque type is being reexported from a dependency.
// Regression test for <https://github.com/rust-lang/rust/issues/160107>.

//@ aux-build: opaque.rs

#![crate_name = "foo"]

extern crate opaque;
pub use opaque::*;
