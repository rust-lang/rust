// compile-flags: --no-defaults

#![crate_name = "foo"]

// @has 'foo/index.html' '//code' 'extern crate std;'
// @!has 'foo/index.html' '//code' 'use std::prelude::v1::*;'
pub struct Foo;
