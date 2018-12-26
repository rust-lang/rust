// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro]
pub fn m(a: TokenStream) -> TokenStream {
    a
}

#[proc_macro_attribute]
pub fn a(_a: TokenStream, b: TokenStream) -> TokenStream {
    b
}
