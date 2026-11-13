extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn declare_big_function(_input: TokenStream) -> TokenStream {
    include_str!("./generated.rs").parse().unwrap()
}
