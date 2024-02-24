//@ force-host
//@ no-prefer-dynamic
//@ compile-flags: --crate-type proc-macro

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Foo)]
pub fn foo(input: TokenStream) -> TokenStream {
    input
}
