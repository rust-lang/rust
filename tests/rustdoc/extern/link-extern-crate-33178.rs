//@ aux-build:empty.rs
//@ aux-build:variant-struct.rs
//@ build-aux-docs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/33178
#![crate_name="issue_33178"]

//@ has issue_33178/index.html
//@ has - '//a[@title="mod empty"][@href="../empty/index.html"]' empty
pub extern crate empty;

//@ has - '//a[@title="mod variant_struct"][@href="../variant_struct/index.html"]' variant_struct
pub extern crate variant_struct as foo;

//@ has - '//a[@title="mod issue_33178"][@href="index.html"]' self
pub extern crate self as bar;
