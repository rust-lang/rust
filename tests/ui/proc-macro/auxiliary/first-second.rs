extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Group, Delimiter};

#[proc_macro_attribute]
pub fn first(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let tokens: TokenStream = "#[derive(Second)]".parse().unwrap();
    let wrapped = TokenTree::Group(Group::new(Delimiter::None, item.into_iter().collect()));
    tokens.into_iter().chain(std::iter::once(wrapped)).collect()
}

#[proc_macro_derive(Second)]
pub fn second(item: TokenStream) -> TokenStream {
    TokenStream::new()
}
