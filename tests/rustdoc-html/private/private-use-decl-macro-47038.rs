#![feature(decl_macro)]

#![crate_name = "foo"]

// https://github.com/rust-lang/rust/issues/47038

use std::vec;

//@ has 'foo/index.html'
//@ !has - '//*[@id="macros"]' 'Macros'
//@ !has - '//a/@href' 'macro.vec.html'
//@ !has 'foo/macro.vec.html'
