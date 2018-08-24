// aux-build:issue-33113.rs
// build-aux-docs
// ignore-cross-compile

extern crate bar;

// @has issue_33113/trait.Bar.html
// @has - '//code' "for &'a char"
// @has - '//code' "for Foo"
pub use bar::Bar;
