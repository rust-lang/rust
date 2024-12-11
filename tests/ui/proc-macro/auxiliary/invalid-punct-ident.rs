#![feature(proc_macro_raw_ident)]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn invalid_punct(_: TokenStream) -> TokenStream {
    TokenTree::from(Punct::new('`', Spacing::Alone)).into()
}

#[proc_macro]
pub fn invalid_ident(_: TokenStream) -> TokenStream {
    TokenTree::from(Ident::new("*", Span::call_site())).into()
}

#[proc_macro]
pub fn invalid_raw_ident(_: TokenStream) -> TokenStream {
    TokenTree::from(Ident::new_raw("self", Span::call_site())).into()
}

#[proc_macro]
pub fn lexer_failure(_: TokenStream) -> TokenStream {
    "a b ) c".parse().expect("parsing failed without panic")
}
