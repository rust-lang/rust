extern crate proc_macro;
use std::iter::FromIterator;
use proc_macro::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};

#[proc_macro]
pub fn make_struct(input: TokenStream) -> TokenStream {
    match input.into_iter().next().unwrap() {
        TokenTree::Ident(ident) => {
            vec![
                TokenTree::Ident(Ident::new("struct", Span::call_site())),
                TokenTree::Ident(Ident::new_raw(&ident.to_string(), Span::call_site())),
                TokenTree::Punct(Punct::new(';', Spacing::Alone))
            ].into_iter().collect()
        }
        _ => panic!()
    }
}

#[proc_macro]
pub fn make_bad_struct(input: TokenStream) -> TokenStream {
    match input.into_iter().next().unwrap() {
        TokenTree::Ident(ident) => {
            vec![
                TokenTree::Ident(Ident::new_raw("struct", Span::call_site())),
                TokenTree::Ident(Ident::new(&ident.to_string(), Span::call_site())),
                TokenTree::Punct(Punct::new(';', Spacing::Alone))
            ].into_iter().collect()
        }
        _ => panic!()
    }
}

#[proc_macro]
pub fn make_raw_module(input: TokenStream) -> TokenStream {
    let TokenTree::Ident(ident) = input.into_iter().next().unwrap() else {
        panic!()
    };
    let name = ident.to_string();
    let name = name.strip_prefix("r#").unwrap();
    let ident = Ident::new_raw(name, Span::call_site());
    let body = TokenStream::from_iter([
        TokenTree::Ident(Ident::new("pub", Span::call_site())),
        TokenTree::Ident(Ident::new("struct", Span::call_site())),
        TokenTree::Ident(Ident::new("Marker", Span::call_site())),
        TokenTree::Punct(Punct::new(';', Spacing::Alone)),
    ]);
    TokenStream::from_iter([
        TokenTree::Ident(Ident::new("mod", Span::call_site())),
        TokenTree::Ident(ident),
        TokenTree::Group(Group::new(Delimiter::Brace, body)),
    ])
}
