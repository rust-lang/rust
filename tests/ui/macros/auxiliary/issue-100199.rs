#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::{quote, Ident, Span, TokenStream, TokenTree};

#[proc_macro_attribute]
pub fn struct_with_bound(_: TokenStream, _: TokenStream) -> TokenStream {
    let crate_ident = TokenTree::Ident(Ident::new("crate", Span::call_site()));
    let trait_ident = TokenTree::Ident(Ident::new("MyTrait", Span::call_site()));
    quote!(
        struct Foo<T: $crate_ident::$trait_ident> {}
    )
}
