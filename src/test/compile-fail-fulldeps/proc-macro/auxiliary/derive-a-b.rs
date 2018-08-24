// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(A)]
pub fn derive_a(_: TokenStream) -> TokenStream {
    "".parse().unwrap()
}

#[proc_macro_derive(B)]
pub fn derive_b(_: TokenStream) -> TokenStream {
    "".parse().unwrap()
}
