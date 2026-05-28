extern crate proc_macro;
use proc_macro::Delimiter::{Brace, Bracket, Parenthesis};
use proc_macro::Spacing::{Alone, Joint};
use proc_macro::{Group, Ident, Punct, Span, TokenStream, TokenTree as TT};

#[proc_macro_derive(Derive)]
pub fn derive(_: TokenStream) -> TokenStream {
    TokenStream::from_iter([
        TT::from(Punct::new('#', Alone)),
        TT::from(Group::new(
            Bracket,
            TokenStream::from_iter([
                TT::from(Ident::new("allow", Span::call_site())),
                TT::from(Group::new(
                    Parenthesis,
                    TokenStream::from_iter([
                        TT::from(Ident::new("clippy", Span::call_site())),
                        TT::from(Punct::new(':', Joint)),
                        TT::from(Punct::new(':', Alone)),
                        TT::from(Ident::new("disallowed_macros", Span::call_site())),
                    ]),
                )),
            ]),
        )),
        TT::from(Ident::new("impl", Span::call_site())),
        TT::from(Ident::new("Foo", Span::call_site())),
        TT::from(Group::new(Brace, TokenStream::new())),
    ])
}
