#![feature(proc_macro_span)]

extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
use std::iter::FromIterator;

/// Expands to a call to `println!` having a format string as its argument
/// but with the format string's span set to that of the proc macro's input token
#[proc_macro]
pub fn foo(input: TokenStream) -> TokenStream {
    let token = input.into_iter().next().unwrap();
    let mut lit: Literal = r#"r"{}""#.parse().unwrap();
    lit.set_span(token.span());
    FromIterator::<TokenTree>::from_iter([
        Ident::new("println", Span::mixed_site()).into(),
        Punct::new('!', Spacing::Alone).into(),
        Group::new(Delimiter::Parenthesis, TokenTree::from(lit).into()).into(),
    ])
}


/// Same as `foo` but with the format string containing multiple `#`s
#[proc_macro]
pub fn foo2(input: TokenStream) -> TokenStream {
    let token = input.into_iter().next().unwrap();
    let mut lit: Literal = r###"r##"{}"##"###.parse().unwrap();
    lit.set_span(token.span());
    FromIterator::<TokenTree>::from_iter([
        Ident::new("println", Span::mixed_site()).into(),
        Punct::new('!', Spacing::Alone).into(),
        Group::new(Delimiter::Parenthesis, TokenTree::from(lit).into()).into(),
    ])
}


/// Same as `foo` but respans to a combination of two consecutive tokens.
/// This makes possible scenarios where
#[proc_macro]
pub fn foo3(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let first = iter.next().unwrap();
    let second = iter.next().unwrap();
    let joined_span = first.span().join(second.span()).unwrap();
    let mut lit: Literal = r#"r"{}""#.parse().unwrap();
    lit.set_span(joined_span);
    FromIterator::<TokenTree>::from_iter([
        Ident::new("println", Span::mixed_site()).into(),
        Punct::new('!', Spacing::Alone).into(),
        Group::new(Delimiter::Parenthesis, TokenTree::from(lit).into()).into(),
    ])
}
