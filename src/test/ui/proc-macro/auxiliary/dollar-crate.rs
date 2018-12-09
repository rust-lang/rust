// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn normalize(input: TokenStream) -> TokenStream {
    input.into_iter().collect()
}
