#![feature(proc_macro_set_group_span)]

extern crate proc_macro;

use std::iter::FromIterator;

use proc_macro::{Delimiter, Group, Ident, Span, TokenStream, TokenTree};

#[proc_macro]
pub fn different_span(args: TokenStream) -> TokenStream {
    let input_tts = args.into_iter().collect::<Vec<_>>();
    assert_eq!(input_tts.len(), 1);

    let TokenTree::Group(input_group) = &input_tts[0] else {
        panic!("must be a group");
    };

    let mut group = Group::new(
        Delimiter::Brace,
        TokenStream::from_iter(vec![TokenTree::Ident(Ident::new("let", Span::call_site()))]),
    );
    group.set_span_open(input_group.span_open());
    group.set_span_close(input_group.span_close());

    let mut out = TokenStream::new();
    out.extend([group]);
    out
}

#[proc_macro]
pub fn same_span(args: TokenStream) -> TokenStream {
    let input_tts = args.into_iter().collect::<Vec<_>>();
    assert_eq!(input_tts.len(), 1);

    let TokenTree::Group(input_group) = &input_tts[0] else {
        panic!("must be a group");
    };

    let mut group = Group::new(
        Delimiter::Brace,
        TokenStream::from_iter(vec![TokenTree::Ident(Ident::new("let", Span::call_site()))]),
    );
    group.set_span(input_group.span());

    let mut out = TokenStream::new();
    out.extend([group]);
    out
}
