// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(A)]
pub fn derive_a(_input: TokenStream) -> TokenStream {
    "struct A { inner }".parse().unwrap()
}
