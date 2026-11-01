extern crate proc_macro;

use proc_macro::*;

#[proc_macro_attribute]
pub fn proc_macro_attribute_that_generates_repr_packed(
    _attr: TokenStream,
    item: TokenStream,
) -> TokenStream {
    let repr = vec![TokenTree::Ident(Ident::new("packed", Span::call_site()))].into_iter();
    let attr = vec![
        TokenTree::Ident(Ident::new("repr", Span::call_site())),
        TokenTree::Group(Group::new(Delimiter::Parenthesis, repr.collect())),
    ]
    .into_iter();
    vec![
        TokenTree::Punct(Punct::new('#', Spacing::Alone)),
        TokenTree::Group(Group::new(Delimiter::Bracket, attr.collect())),
    ]
    .into_iter()
    .chain(item)
    .collect()
}
