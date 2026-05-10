extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn skip(_: TokenStream, input: TokenStream) -> TokenStream {
    "compile_error! { \"x\" }".parse().unwrap()
}
