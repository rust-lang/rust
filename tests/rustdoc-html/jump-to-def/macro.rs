//@ aux-build:symbols.rs
//@ build-aux-docs
//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/macro.rs.html'

#[macro_use]
extern crate symbols;

//@ has - '//a[@href="../../symbols/macro.symbols.html"]' 'symbols!'
symbols! {
    A = 12
}
