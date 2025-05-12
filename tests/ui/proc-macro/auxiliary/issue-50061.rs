extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn check(_a: TokenStream, b: TokenStream) -> TokenStream {
    b.into_iter().collect()
}
