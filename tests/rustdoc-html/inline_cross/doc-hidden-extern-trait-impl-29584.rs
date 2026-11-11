//@ aux-build:issue-29584.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/29584
#![crate_name="issue_29584"]

extern crate issue_29584;

//@ has issue_29584/struct.Foo.html
//@ !hasraw - 'impl Bar for'
pub use issue_29584::Foo;
