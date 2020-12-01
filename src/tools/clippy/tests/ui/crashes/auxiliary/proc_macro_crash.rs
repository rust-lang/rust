// compile-flags: --emit=link
// no-prefer-dynamic
// ^ compiletest by default builds all aux files as dylibs, but we don't want that for proc-macro
// crates. If we don't set this, compiletest will override the `crate_type` attribute below and
// compile this as dylib. Removing this then causes the test to fail because a `dylib` crate can't
// contain a proc-macro.

#![feature(repr128)]
#![allow(incomplete_features)]
#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Span, TokenStream, TokenTree};
use std::iter::FromIterator;

#[proc_macro]
pub fn macro_test(input_stream: TokenStream) -> TokenStream {
    let first_token = input_stream.into_iter().next().unwrap();
    let span = first_token.span();

    TokenStream::from_iter(vec![
        TokenTree::Ident(Ident::new("fn", Span::call_site())),
        TokenTree::Ident(Ident::new("code", Span::call_site())),
        TokenTree::Group(Group::new(Delimiter::Parenthesis, TokenStream::new())),
        TokenTree::Group(Group::new(Delimiter::Brace, {
            let mut clause = Group::new(Delimiter::Brace, TokenStream::new());
            clause.set_span(span);

            TokenStream::from_iter(vec![
                TokenTree::Ident(Ident::new("if", Span::call_site())),
                TokenTree::Ident(Ident::new("true", Span::call_site())),
                TokenTree::Group(clause.clone()),
                TokenTree::Ident(Ident::new("else", Span::call_site())),
                TokenTree::Group(clause),
            ])
        })),
    ])
}
