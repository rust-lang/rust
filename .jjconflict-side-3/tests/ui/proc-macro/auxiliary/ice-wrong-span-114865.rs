#![feature(proc_macro_span)]

extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
use std::iter::FromIterator;

/// Builds a `println!(<fmt_str>)` token stream with the given span on the format string literal.
fn make_println(fmt_str: &str, span: Span) -> TokenStream {
    let mut lit: Literal = fmt_str.parse().unwrap();
    lit.set_span(span);
    FromIterator::<TokenTree>::from_iter([
        Ident::new("println", Span::mixed_site()).into(),
        Punct::new('!', Spacing::Alone).into(),
        Group::new(Delimiter::Parenthesis, TokenTree::from(lit).into()).into(),
    ])
}

/// Expands to `println!(r"{}")` with the span of the first input token.
#[proc_macro]
pub fn foo(input: TokenStream) -> TokenStream {
    let span = input.into_iter().next().unwrap().span();
    make_println(r#"r"{}""#, span)
}

/// Same as `foo` but with hashes: expands to `println!(r##"{}"##)`.
#[proc_macro]
pub fn foo2(input: TokenStream) -> TokenStream {
    let span = input.into_iter().next().unwrap().span();
    make_println(r###"r##"{}"##"###, span)
}

/// Expands to `println!(r"{}")` with a span joining two input tokens,
/// creating a span whose source text may not be a valid raw string.
#[proc_macro]
pub fn foo3(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let span = iter.next().unwrap().span().join(iter.next().unwrap().span()).unwrap();
    make_println(r#"r"{}""#, span)
}

/// Same as `foo3` but with hashes: expands to `println!(r##"{}"##)`.
#[proc_macro]
pub fn foo4(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let span = iter.next().unwrap().span().join(iter.next().unwrap().span()).unwrap();
    make_println(r###"r##"{}"##"###, span)
}
