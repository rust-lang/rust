// This test ensures that even when a proc-macro is used, the `Source` link points
// to the right file. In this test, the `Source` link we're testing is located in
// the `attribute_macro_path_demo` module.
// Regression test for <https://github.com/rust-lang/rust/issues/158768>.

//@ aux-build: attribute-macro-path.rs

#![crate_name = "foo"]
#![feature(custom_inner_attributes, proc_macro_hygiene)]

extern crate attribute_macro_path;

//@ has 'foo/attribute_macro_path_demo/index.html'
//@ has - '//a[@href="../../src/foo/attribute-macro-path.rs.html#16"]' 'Source'

pub mod attribute_macro_path_demo;
