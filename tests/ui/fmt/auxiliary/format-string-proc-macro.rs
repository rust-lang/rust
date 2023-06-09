// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
use std::iter::FromIterator;

#[proc_macro]
pub fn foo_with_input_span(input: TokenStream) -> TokenStream {
    let span = input.into_iter().next().unwrap().span();

    let mut lit = Literal::string("{foo}");
    lit.set_span(span);

    TokenStream::from(TokenTree::Literal(lit))
}

#[proc_macro]
pub fn err_with_input_span(input: TokenStream) -> TokenStream {
    let span = input.into_iter().next().unwrap().span();

    let mut lit = Literal::string("         }");
    lit.set_span(span);

    TokenStream::from(TokenTree::Literal(lit))
}

fn build_format(args: impl Into<TokenStream>) -> TokenStream {
    TokenStream::from_iter([
        TokenTree::from(Ident::new("format", Span::call_site())),
        TokenTree::from(Punct::new('!', Spacing::Alone)),
        TokenTree::from(Group::new(Delimiter::Parenthesis, args.into())),
    ])
}

#[proc_macro]
pub fn respan_to_invalid_format_literal(input: TokenStream) -> TokenStream {
    let mut s = Literal::string("{");
    s.set_span(input.into_iter().next().unwrap().span());

    build_format(TokenTree::from(s))
}

#[proc_macro]
pub fn capture_a_with_prepended_space_preserve_span(input: TokenStream) -> TokenStream {
    let mut s = Literal::string(" {a}");
    s.set_span(input.into_iter().next().unwrap().span());

    build_format(TokenTree::from(s))
}

#[proc_macro]
pub fn format_args_captures(_: TokenStream) -> TokenStream {
    r#"{ let x = 5; format!("{x}") }"#.parse().unwrap()
}

#[proc_macro]
pub fn bad_format_args_captures(_: TokenStream) -> TokenStream {
    r#"{ let x = 5; format!(concat!("{x}")) }"#.parse().unwrap()
}

#[proc_macro]
pub fn identity_pm(input: TokenStream) -> TokenStream {
    input
}
