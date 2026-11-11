//@ edition: 2018
extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(RustEmbed)]
pub fn rust_embed_derive(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}
