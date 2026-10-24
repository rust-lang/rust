//@ edition:2024

extern crate proc_macro;

use proc_macro::TokenStream;
use std::str::FromStr;

#[proc_macro]
pub fn number_of_tokens_in_a_guarded_string_literal(_: TokenStream) -> TokenStream {
    TokenStream::from_str("#\"abc\"#").unwrap().into_iter().count().to_string().parse().unwrap()
}

#[proc_macro]
pub fn number_of_tokens_in_a_guarded_unterminated_string_literal(_: TokenStream) -> TokenStream {
    TokenStream::from_str("#\"abc\"").unwrap().into_iter().count().to_string().parse().unwrap()
}
