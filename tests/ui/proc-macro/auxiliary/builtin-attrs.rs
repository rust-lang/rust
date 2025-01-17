extern crate proc_macro;
use proc_macro::*;

#[proc_macro_attribute]
pub fn feature(_: TokenStream, input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_attribute]
pub fn repr(_: TokenStream, input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_attribute]
pub fn test(_: TokenStream, input: TokenStream) -> TokenStream {
    "struct Test;".parse().unwrap()
}

#[proc_macro_attribute]
pub fn bench(_: TokenStream, input: TokenStream) -> TokenStream {
    "struct Bench;".parse().unwrap()
}
