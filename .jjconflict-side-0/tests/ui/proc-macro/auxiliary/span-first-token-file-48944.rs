extern crate proc_macro;

use proc_macro::TokenStream;

// Parses a multi-token string with `FromStr` and checks that every token reports the same
// source file. The first token used to get a different file than the rest.
#[proc_macro]
pub fn check_first_token_file(_: TokenStream) -> TokenStream {
    let tokens: Vec<_> = "aaa\nbbb".parse::<TokenStream>().unwrap().into_iter().collect();
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].span().file(), tokens[1].span().file());
    TokenStream::new()
}
