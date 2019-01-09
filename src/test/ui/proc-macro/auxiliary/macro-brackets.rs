// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro_attribute]
pub fn doit(_: TokenStream, input: TokenStream) -> TokenStream {
    input.into_iter().collect()
}
