extern crate proc_macro;
use proc_macro::{TokenStream, TokenTree};

#[proc_macro_derive(Empty2, attributes(empty_helper))]
pub fn empty_derive2(_: TokenStream) -> TokenStream {
    TokenStream::new()
}
