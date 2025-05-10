extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Span, TokenStream, TokenTree};

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
