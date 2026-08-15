//@ check-pass

#![warn(clippy::missing_inline_in_public_items)]

extern crate proc_macro;

use proc_macro::TokenStream;

fn _foo() {}

#[proc_macro]
pub fn function_like(_: TokenStream) -> TokenStream {
    TokenStream::new()
}

#[proc_macro_attribute]
pub fn attribute(_: TokenStream, _: TokenStream) -> TokenStream {
    TokenStream::new()
}

#[proc_macro_derive(Derive)]
pub fn derive(_: TokenStream) -> TokenStream {
    TokenStream::new()
}
