//@ compile-flags: -Z unstable-options --document-hidden-items --document-private-items

#![crate_name = "foo"]

//@ !has 'foo/index.html' '//code' 'extern crate std;'
//@ !has 'foo/index.html' '//code' 'use std::prelude'
pub struct Foo;
