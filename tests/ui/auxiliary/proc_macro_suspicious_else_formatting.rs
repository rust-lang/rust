// compile-flags: --emit=link
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::{token_stream, Delimiter, Group, Ident, Span, TokenStream, TokenTree};
use std::iter::FromIterator;

fn read_ident(iter: &mut token_stream::IntoIter) -> Ident {
    match iter.next() {
        Some(TokenTree::Ident(i)) => i,
        _ => panic!("expected ident"),
    }
}

#[proc_macro_derive(DeriveBadSpan)]
pub fn derive_bad_span(input: TokenStream) -> TokenStream {
    let mut input = input.into_iter();
    assert_eq!(read_ident(&mut input).to_string(), "struct");
    let ident = read_ident(&mut input);
    let mut tys = match input.next() {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => g.stream().into_iter(),
        _ => panic!(),
    };
    let field1 = read_ident(&mut tys);
    tys.next();
    let field2 = read_ident(&mut tys);

    <TokenStream as FromIterator<TokenTree>>::from_iter(
        [
            Ident::new("impl", Span::call_site()).into(),
            ident.into(),
            Group::new(
                Delimiter::Brace,
                <TokenStream as FromIterator<TokenTree>>::from_iter(
                    [
                        Ident::new("fn", Span::call_site()).into(),
                        Ident::new("_foo", Span::call_site()).into(),
                        Group::new(Delimiter::Parenthesis, TokenStream::new()).into(),
                        Group::new(
                            Delimiter::Brace,
                            <TokenStream as FromIterator<TokenTree>>::from_iter(
                                [
                                    Ident::new("if", field1.span()).into(),
                                    Ident::new("true", field1.span()).into(),
                                    {
                                        let mut group = Group::new(Delimiter::Brace, TokenStream::new());
                                        group.set_span(field1.span());
                                        group.into()
                                    },
                                    Ident::new("if", field2.span()).into(),
                                    Ident::new("true", field2.span()).into(),
                                    {
                                        let mut group = Group::new(Delimiter::Brace, TokenStream::new());
                                        group.set_span(field2.span());
                                        group.into()
                                    },
                                ]
                                .iter()
                                .cloned(),
                            ),
                        )
                        .into(),
                    ]
                    .iter()
                    .cloned(),
                ),
            )
            .into(),
        ]
        .iter()
        .cloned(),
    )
}
