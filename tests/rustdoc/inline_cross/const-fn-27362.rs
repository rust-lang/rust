//@ aux-build:issue-27362-aux.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/27362
#![crate_name="issue_27362"]

extern crate issue_27362_aux;

pub use issue_27362_aux::*;

//@ matches issue_27362/fn.foo.html '//pre' "pub const fn foo()"
//@ matches issue_27362/fn.bar.html '//pre' "pub const unsafe fn bar()"
//@ matches issue_27362/struct.Foo.html '//h4[@class="code-header"]' "const unsafe fn baz()"
