//@ aux-build:issue-106421-force-unstable.rs
//@ ignore-cross-compile
// This is the version where a non-compiler-internal crate inlines a compiler-internal one.
// In this case, the item shouldn't be documented, because regular users can't get at it.
extern crate foo;

// @!has issue_106421_not_internal/struct.FatalError.html '//*[@id="method.raise"]' 'fn raise'
pub use foo::FatalError;
