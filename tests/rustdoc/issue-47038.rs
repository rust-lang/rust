#![feature(decl_macro)]

#![crate_name = "foo"]

use std::vec;

// @has 'foo/index.html'
// @!has - '//*[@id="macros"]' 'Macros'
// @!has - '//a/@href' 'macro.vec.html'
// @!has 'foo/macro.vec.html'
