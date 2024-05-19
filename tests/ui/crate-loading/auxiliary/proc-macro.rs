//@ force-host
//@ no-prefer-dynamic
#![crate_name = "reproduction"]
#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn mac(input: TokenStream) -> TokenStream {
    input
}
