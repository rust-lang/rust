extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};

// Re-emits each enum variant's identifier as an associated constant, reusing the
// original `Ident` tokens so the generated items keep the variants' spans.
#[proc_macro_derive(Bar)]
pub fn derive_bar(input: TokenStream) -> TokenStream {
    let mut it = input.into_iter();
    let mut name: Option<Ident> = None;
    let mut body: Option<Group> = None;

    while let Some(tt) = it.next() {
        match tt {
            TokenTree::Ident(id) => {
                if id.to_string() == "enum" {
                    if let Some(TokenTree::Ident(n)) = it.next() {
                        name = Some(n);
                    }
                }
            }
            TokenTree::Group(g) => {
                if g.delimiter() == Delimiter::Brace {
                    body = Some(g);
                    break;
                }
            }
            _ => {}
        }
    }

    let name = name.expect("enum name");
    let body = body.expect("enum body");

    // Collect variant idents (skip commas / discriminants).
    let mut variants: Vec<Ident> = Vec::new();
    let mut expect_ident = true;
    for tt in body.stream() {
        match tt {
            TokenTree::Ident(id) => {
                if expect_ident {
                    variants.push(id);
                    expect_ident = false;
                }
            }
            TokenTree::Punct(p) => {
                if p.as_char() == ',' {
                    expect_ident = true;
                }
            }
            _ => {}
        }
    }

    // Build: impl Name { const V: () = (); ... }
    let mut inner: Vec<TokenTree> = Vec::new();
    for v in variants {
        inner.push(TokenTree::Ident(Ident::new("const", Span::call_site())));
        inner.push(TokenTree::Ident(v)); // original ident + span
        inner.push(TokenTree::Punct(Punct::new(':', Spacing::Alone)));
        inner.push(TokenTree::Group(Group::new(Delimiter::Parenthesis, TokenStream::new())));
        inner.push(TokenTree::Punct(Punct::new('=', Spacing::Alone)));
        inner.push(TokenTree::Group(Group::new(Delimiter::Parenthesis, TokenStream::new())));
        inner.push(TokenTree::Punct(Punct::new(';', Spacing::Alone)));
    }

    let mut out: Vec<TokenTree> = Vec::new();
    out.push(TokenTree::Ident(Ident::new("impl", Span::call_site())));
    out.push(TokenTree::Ident(name));
    out.push(TokenTree::Group(Group::new(Delimiter::Brace, inner.into_iter().collect())));

    out.into_iter().collect()
}
