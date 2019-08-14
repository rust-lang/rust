// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Test)]
pub fn derive(_input: TokenStream) -> TokenStream {
    "fn f(s: S) { s.x }".parse().unwrap()
}

#[proc_macro_attribute]
pub fn attr_test(_attr: TokenStream, input: TokenStream) -> TokenStream {
    input
}
