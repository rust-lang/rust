extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(A)]
pub fn derive(input: TokenStream) -> TokenStream {
    let input = input.to_string();
    assert!(input.contains("struct A;"));
    "".parse().unwrap()
}
