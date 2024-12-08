extern crate proc_macro;

use proc_macro::*;

#[proc_macro]
pub fn single_quote_alone(_: TokenStream) -> TokenStream {
    // `&'a u8`, but the `'` token is not joint
    let trees: Vec<TokenTree> = vec![
        Punct::new('&', Spacing::Alone).into(),
        Punct::new('\'', Spacing::Alone).into(),
        Ident::new("a", Span::call_site()).into(),
        Ident::new("u8", Span::call_site()).into(),
    ];
    trees.into_iter().collect()
}
