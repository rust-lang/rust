extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, TokenStream, TokenTree};

#[proc_macro]
pub fn unsafe_block(input: TokenStream) -> TokenStream {
    let span = input.into_iter().next().unwrap().span();
    TokenStream::from_iter([TokenTree::Ident(Ident::new("unsafe", span)), {
        let mut group = Group::new(Delimiter::Brace, TokenStream::new());
        group.set_span(span);
        TokenTree::Group(group)
    }])
}
