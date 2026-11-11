extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(B, attributes(B))]
pub fn derive_b(input: TokenStream) -> TokenStream {
    "".parse().unwrap()
}
