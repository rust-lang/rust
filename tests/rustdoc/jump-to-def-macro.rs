//@ aux-build:jump-to-def-macro.rs
//@ build-aux-docs
//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/jump-to-def-macro.rs.html'

#[macro_use]
extern crate jump_to_def_macro;

//@ has - '//a[@href="../../jump_to_def_macro/macro.symbols.html"]' 'symbols!'
symbols! {
    A = 12
}
