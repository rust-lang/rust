//@ check-pass
//@ force-host
//@ no-prefer-dynamic
//@ compile-flags: -Z unpretty=expanded
//@ needs-unwind compiling proc macros with panic=abort causes a warning
//@ edition: 2015

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn a(x: TokenStream) -> TokenStream {
    x
}

#[proc_macro_derive(B, attributes(attr_a, attr_b))]
pub fn b(x: TokenStream) -> TokenStream {
    x
}

#[proc_macro_attribute]
pub fn c(x: TokenStream, y: TokenStream) -> TokenStream {
    y
}
