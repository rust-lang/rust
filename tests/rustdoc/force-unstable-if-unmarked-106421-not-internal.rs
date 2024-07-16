//@ aux-build:issue-106421-force-unstable.rs
//@ ignore-cross-compile
// This is the version where a non-compiler-internal crate inlines a compiler-internal one.
// In this case, the item shouldn't be documented, because regular users can't get at it.
// https://github.com/rust-lang/rust/issues/106421
#![crate_name="bar"]

extern crate foo;

//@ !has bar/struct.FatalError.html '//*[@id="method.raise"]' 'fn raise'
pub use foo::FatalError;
