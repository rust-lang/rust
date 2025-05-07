extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn attr(_args: TokenStream, input: TokenStream) -> TokenStream {
    println!("{}", input);
    TokenStream::new()
}
