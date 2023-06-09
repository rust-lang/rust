// aux-build:issue-36031.rs
// build-aux-docs
// ignore-cross-compile

#![crate_name = "foo"]

extern crate issue_36031;

pub use issue_36031::Foo;
