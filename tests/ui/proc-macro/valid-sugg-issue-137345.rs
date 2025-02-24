//@ aux-crate: quote=quote-issue-137345.rs

extern crate proc_macro;
extern crate quote;

use proc_macro::TokenStream;

pub fn default_args_fn(_: TokenStream) -> TokenStream {
    let decl_macro = TokenStream::new();

    quote::quote! {
        #(#decl_macro)*
    }
    .into() //~^^^ ERROR no method named `map` found for opaque type
}

fn main() {}
