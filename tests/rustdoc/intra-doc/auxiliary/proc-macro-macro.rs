// force-host
// no-prefer-dynamic
// compile-flags: --crate-type proc-macro

#![crate_type="proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(DeriveA)]
pub fn a_derive(input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_derive(DeriveB)]
pub fn b_derive(input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_derive(DeriveTrait)]
pub fn trait_derive(input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_attribute]
pub fn attr_a(input: TokenStream, _args: TokenStream) -> TokenStream {
    input
}

#[proc_macro_attribute]
pub fn attr_b(input: TokenStream, _args: TokenStream) -> TokenStream {
    input
}
