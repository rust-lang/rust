//@ aux-build:issue-99221-aux.rs
//@ build-aux-docs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/99221
#![crate_name = "foo"]

#[macro_use]
extern crate issue_99221_aux;

pub use issue_99221_aux::*;

//@ count foo/index.html '//a[@class="struct"][@title="struct foo::Print"]' 1

pub struct Print;
