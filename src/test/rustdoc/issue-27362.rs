// aux-build:issue-27362-aux.rs
// ignore-cross-compile

extern crate issue_27362_aux;

pub use issue_27362_aux::*;

// @matches issue_27362/fn.foo.html '//pre' "pub const fn foo()"
// @matches issue_27362/fn.bar.html '//pre' "pub const unsafe fn bar()"
// @matches issue_27362/struct.Foo.html '//code' "const unsafe fn baz()"
