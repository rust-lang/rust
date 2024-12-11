//#![feature(proc_macro_diagnostic, proc_macro_span, proc_macro_def_site)]

extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use std::iter::FromIterator;

#[proc_macro_attribute]
pub fn repro(_args: TokenStream, input: TokenStream) -> TokenStream {
    let call_site = Span::call_site();
    let span = input.into_iter().nth(8).unwrap().span();

    //fn f(_: &::std::fmt::Formatter) {}
    TokenStream::from_iter([
        TokenTree::Ident(Ident::new("fn", call_site)),
        TokenTree::Ident(Ident::new("f", call_site)),
        TokenTree::Group(Group::new(
            Delimiter::Parenthesis,
            TokenStream::from_iter([
                TokenTree::Ident(Ident::new("_", call_site)),
                TokenTree::Punct(punct(':', Spacing::Alone, call_site)),
                TokenTree::Punct(punct('&', Spacing::Alone, call_site)),
                TokenTree::Punct(punct(':', Spacing::Joint, span)),
                TokenTree::Punct(punct(':', Spacing::Alone, span)),
                TokenTree::Ident(Ident::new("std", span)),
                TokenTree::Punct(punct(':', Spacing::Joint, span)),
                TokenTree::Punct(punct(':', Spacing::Alone, span)),
                TokenTree::Ident(Ident::new("fmt", span)),
                TokenTree::Punct(punct(':', Spacing::Joint, span)),
                TokenTree::Punct(punct(':', Spacing::Alone, span)),
                TokenTree::Ident(Ident::new("Formatter", span)),
            ]),
        )),
        TokenTree::Group(Group::new(Delimiter::Brace, TokenStream::new())),
    ])
}

fn punct(ch: char, spacing: Spacing, span: Span) -> Punct {
    let mut punct = Punct::new(ch, spacing);
    punct.set_span(span);
    punct
}
