// aux-build:issue-27362.rs
// ignore-cross-compile
// ignore-test This test fails on beta/stable #32019

extern crate issue_27362;
pub use issue_27362 as quux;

// @matches issue_27362/quux/fn.foo.html '//pre' "pub const fn foo()"
// @matches issue_27362/quux/fn.bar.html '//pre' "pub const unsafe fn bar()"
// @matches issue_27362/quux/struct.Foo.html '//code' "const unsafe fn baz()"
