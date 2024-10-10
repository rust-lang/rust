//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_quote)]
#![feature(proc_macro_totokens)]

extern crate proc_macro;

use proc_macro::{Ident, Span, TokenStream, TokenTree, quote};

#[proc_macro_attribute]
pub fn struct_with_bound(_: TokenStream, _: TokenStream) -> TokenStream {
    let crate_ident = TokenTree::Ident(Ident::new("crate", Span::call_site()));
    let trait_ident = TokenTree::Ident(Ident::new("MyTrait", Span::call_site()));
    quote!(
        struct Foo<T: $crate_ident::$trait_ident> {}
    )
}
