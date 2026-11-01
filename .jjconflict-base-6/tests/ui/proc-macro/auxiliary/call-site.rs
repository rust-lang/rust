extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn check(input: TokenStream) -> TokenStream {
    // Parsed `x2` can refer to `x2` from `input`
    let parsed1: TokenStream = "let x3 = x2;".parse().unwrap();
    // `x3` parsed from one string can refer to `x3` parsed from another string.
    let parsed2: TokenStream = "let x4 = x3;".parse().unwrap();
    // Manually assembled `x4` can refer to parsed `x4`.
    let manual: Vec<TokenTree> = vec![
        Ident::new("let", Span::call_site()).into(),
        Ident::new("x5", Span::call_site()).into(),
        Punct::new('=', Spacing::Alone).into(),
        Ident::new("x4", Span::call_site()).into(),
        Punct::new(';', Spacing::Alone).into(),
    ];
    input.into_iter().chain(parsed1.into_iter())
                     .chain(parsed2.into_iter())
                     .chain(manual.into_iter())
                     .collect()
}
