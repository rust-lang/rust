//@ aux-build:issue-99734-aux.rs
//@ build-aux-docs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/99734
#![crate_name = "foo"]

#[macro_use]
extern crate issue_99734_aux;

pub use issue_99734_aux::*;

//@ count foo/index.html '//a[@class="fn"][@title="fn foo::main"]' 1

extern "C" {
    pub fn main() -> std::ffi::c_int;
}
