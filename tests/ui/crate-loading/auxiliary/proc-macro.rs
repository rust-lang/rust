#![crate_name = "reproduction"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn mac(input: TokenStream) -> TokenStream {
    input
}
