//@ compile-flags: --crate-type=proc-macro
//@ has 'foo/index.html' '//a[@href="macro.my_macro.html"]' 'my_macro'
//! Link to [`my_macro`].
#![crate_name = "foo"]

// regression test for https://github.com/rust-lang/rust/issues/91274

extern crate proc_macro;

#[proc_macro]
pub fn my_macro(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    input
}
