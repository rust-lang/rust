extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Nothing)]
pub fn nothing(input: TokenStream) -> TokenStream {
    "".parse().unwrap()
}
