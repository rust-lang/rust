// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn foo(_attr: TokenStream, input: TokenStream) -> TokenStream {
    input.into_iter().collect()
}
