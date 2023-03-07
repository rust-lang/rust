// aux-build:issue-33113.rs
// build-aux-docs
// ignore-cross-compile

extern crate bar;

// @has issue_33113/trait.Bar.html
// @has - '//h3[@class="code-header"]' "for &'a char"
// @has - '//h3[@class="code-header"]' "for Foo"
pub use bar::Bar;
