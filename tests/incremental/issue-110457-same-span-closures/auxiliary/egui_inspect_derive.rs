extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

#[proc_macro]
pub fn expand(_: TokenStream) -> TokenStream {
    // Hand expansion/rewriting of
    // ```
    // quote! {
    //     output_mut(|o| o.copied_text = "".into());
    //     output_mut(|o| o.copied_text = format!("{:?}", self.tile_db));
    // }.into()
    // ```
    stream([
        ident("output_mut"),
        group(
            Delimiter::Parenthesis,
            [
                or(),
                ident("o"),
                or(),
                ident("o"),
                dot(),
                ident("copied_text"),
                eq(),
                string(""),
                dot(),
                ident("into"),
                group(Delimiter::Parenthesis, []),
            ],
        ),
        semi(),
        ident("output_mut"),
        group(
            Delimiter::Parenthesis,
            [
                or(),
                ident("o"),
                or(),
                ident("o"),
                dot(),
                ident("copied_text"),
                eq(),
                ident("format"),
                bang(),
                group(
                    Delimiter::Parenthesis,
                    [string("{:?}"), comma(), ident("self"), dot(), ident("tile_db")],
                ),
            ],
        ),
        semi(),
    ])
}

fn stream(s: impl IntoIterator<Item = TokenTree>) -> TokenStream {
    s.into_iter().collect()
}

fn ident(i: &str) -> TokenTree {
    TokenTree::Ident(Ident::new(i, Span::call_site()))
}
fn group(d: Delimiter, s: impl IntoIterator<Item = TokenTree>) -> TokenTree {
    TokenTree::Group(Group::new(d, s.into_iter().collect()))
}
fn semi() -> TokenTree {
    TokenTree::Punct(Punct::new(';', Spacing::Alone))
}
fn or() -> TokenTree {
    TokenTree::Punct(Punct::new('|', Spacing::Alone))
}
fn dot() -> TokenTree {
    TokenTree::Punct(Punct::new('.', Spacing::Alone))
}
fn eq() -> TokenTree {
    TokenTree::Punct(Punct::new('=', Spacing::Alone))
}
fn bang() -> TokenTree {
    TokenTree::Punct(Punct::new('!', Spacing::Alone))
}
fn comma() -> TokenTree {
    TokenTree::Punct(Punct::new(',', Spacing::Alone))
}
fn string(s: &str) -> TokenTree {
    TokenTree::Literal(Literal::string(s))
}
