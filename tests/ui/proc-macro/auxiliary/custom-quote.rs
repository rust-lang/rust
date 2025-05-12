// ignore-tidy-linelength

#![feature(proc_macro_quote)]

extern crate proc_macro;
use std::iter::FromIterator;
use std::str::FromStr;
use proc_macro::*;

#[proc_macro]
pub fn custom_quote(input: TokenStream) -> TokenStream {
    let mut tokens: Vec<_> = input.into_iter().collect();
    assert_eq!(tokens.len(), 1, "Unexpected input: {:?}", tokens);
    match tokens.pop() {
        Some(TokenTree::Ident(ident)) => {
            assert_eq!(ident.to_string(), "my_ident");

            let proc_macro_crate = TokenStream::from_str("::proc_macro").unwrap();
            let quoted_span = proc_macro::quote_span(proc_macro_crate, ident.span());
            let prefix = TokenStream::from_str(r#"let mut ident = proc_macro::Ident::new("my_ident", proc_macro::Span::call_site());"#).unwrap();
            let set_span_method = TokenStream::from_str("ident.set_span").unwrap();
            let set_span_arg = TokenStream::from(TokenTree::Group(Group::new(Delimiter::Parenthesis, quoted_span)));
            let suffix = TokenStream::from_str(";proc_macro::TokenStream::from(proc_macro::TokenTree::Ident(ident))").unwrap();
            let full_stream = TokenStream::from_iter([prefix, set_span_method, set_span_arg, suffix]);
            full_stream
        }
        _ => unreachable!()
    }
}
