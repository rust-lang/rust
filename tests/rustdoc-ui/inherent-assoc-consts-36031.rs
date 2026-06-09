//@ aux-build:issue-36031.rs
//@ check-pass
//@ build-aux-docs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/36031

#![crate_name = "foo"]

extern crate issue_36031;

pub use issue_36031::Foo;
