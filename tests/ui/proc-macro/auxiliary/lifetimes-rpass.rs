// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro]
pub fn lifetimes_bang(input: TokenStream) -> TokenStream {
    // Roundtrip through token trees
    input.into_iter().collect()
}

#[proc_macro_attribute]
pub fn lifetimes_attr(_: TokenStream, input: TokenStream) -> TokenStream {
    // Roundtrip through AST
    input
}

#[proc_macro_derive(Lifetimes)]
pub fn lifetimes_derive(input: TokenStream) -> TokenStream {
    // Roundtrip through a string
    format!("mod m {{ {} }}", input).parse().unwrap()
}
