//@ aux-build:issue-106421-force-unstable.rs
//@ ignore-cross-compile
//@ compile-flags: -Zforce-unstable-if-unmarked

extern crate foo;

// @has issue_106421/struct.FatalError.html '//*[@id="method.raise"]' 'fn raise'
pub use foo::FatalError;
