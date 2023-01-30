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


#[proc_macro]
pub fn respan_to_invalid_format_literal(input: TokenStream) -> TokenStream {
    let mut s = Literal::string("{");
    s.set_span(input.into_iter().next().unwrap().span());
    TokenStream::from_iter([
        TokenTree::from(Ident::new("format", Span::call_site())),
        TokenTree::from(Punct::new('!', Spacing::Alone)),
        TokenTree::from(Group::new(Delimiter::Parenthesis, TokenTree::from(s).into())),
    ])
}

#[proc_macro]
pub fn capture_a_with_prepended_space_preserve_span(input: TokenStream) -> TokenStream {
    let mut s = Literal::string(" {a}");
    s.set_span(input.into_iter().next().unwrap().span());
    TokenStream::from_iter([
        TokenTree::from(Ident::new("format", Span::call_site())),
        TokenTree::from(Punct::new('!', Spacing::Alone)),
        TokenTree::from(Group::new(Delimiter::Parenthesis, TokenTree::from(s).into())),
    ])
}
