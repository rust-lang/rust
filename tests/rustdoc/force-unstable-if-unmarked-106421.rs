//@ aux-build:issue-106421-force-unstable.rs
//@ ignore-cross-compile
//@ compile-flags: -Zforce-unstable-if-unmarked
// https://github.com/rust-lang/rust/issues/106421
#![crate_name="bar"]

extern crate foo;

//@ has bar/struct.FatalError.html '//*[@id="method.raise"]' 'fn raise'
pub use foo::FatalError;
