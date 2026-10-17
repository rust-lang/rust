extern crate proc_macro;
use proc_macro::{TokenStream, TokenTree, Ident, Punct, Spacing, Span};

#[proc_macro]
pub fn make_struct(input: TokenStream) -> TokenStream {
    match input.into_iter().next().unwrap() {
        TokenTree::Ident(ident) => {
            vec![
                TokenTree::Ident(Ident::new("struct", Span::call_site())),
                TokenTree::Ident(Ident::new_raw(&ident.to_string(), Span::call_site())),
                TokenTree::Punct(Punct::new(';', Spacing::Alone))
            ].into_iter().collect()
        }
        _ => panic!()
    }
}

#[proc_macro]
pub fn make_bad_struct(input: TokenStream) -> TokenStream {
    match input.into_iter().next().unwrap() {
        TokenTree::Ident(ident) => {
            vec![
                TokenTree::Ident(Ident::new_raw("struct", Span::call_site())),
                TokenTree::Ident(Ident::new(&ident.to_string(), Span::call_site())),
                TokenTree::Punct(Punct::new(';', Spacing::Alone))
            ].into_iter().collect()
        }
        _ => panic!()
    }
}
