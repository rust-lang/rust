extern crate proc_macro;

use proc_macro::*;

// This proc macro ignores its input and returns this token stream
//
//   impl <«A1»: Comparable> Comparable for («A1»,) {}
//
// where `«`/`»` are invisible delimiters. This was being misparsed in bug
// #128895.
#[proc_macro]
pub fn main(_input: TokenStream) -> TokenStream {
    let a1 = TokenTree::Group(
        Group::new(
            Delimiter::None,
            std::iter::once(TokenTree::Ident(Ident::new("A1", Span::call_site()))).collect(),
        )
    );
    vec![
        TokenTree::Ident(Ident::new("impl", Span::call_site())),
        TokenTree::Punct(Punct::new('<', Spacing::Alone)),
        a1.clone(),
        TokenTree::Punct(Punct::new(':', Spacing::Alone)),
        TokenTree::Ident(Ident::new("Comparable", Span::call_site())),
        TokenTree::Punct(Punct::new('>', Spacing::Alone)),
        TokenTree::Ident(Ident::new("Comparable", Span::call_site())),
        TokenTree::Ident(Ident::new("for", Span::call_site())),
        TokenTree::Group(
            Group::new(
                Delimiter::Parenthesis,
                vec![
                    a1.clone(),
                    TokenTree::Punct(Punct::new(',', Spacing::Alone)),
                ].into_iter().collect::<TokenStream>(),
            )
        ),
        TokenTree::Group(Group::new(Delimiter::Brace, TokenStream::new())),
    ].into_iter().collect::<TokenStream>()
}
