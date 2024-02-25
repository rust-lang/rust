//@ aux-build:issue-99221-aux.rs
//@ build-aux-docs
//@ ignore-cross-compile

#![crate_name = "foo"]

#[macro_use]
extern crate issue_99221_aux;

pub use issue_99221_aux::*;

// @count foo/index.html '//a[@class="macro"]' 1

#[macro_export]
macro_rules! print {
    () => ()
}
