//@ edition:2018

extern crate proc_macro;

use proc_macro::TokenStream;
use std::str::FromStr;

#[proc_macro]
pub fn number_of_tokens_in_a_prefixed_integer_literal(_: TokenStream) -> TokenStream {
    TokenStream::from_str("hey#123").unwrap().into_iter().count().to_string().parse().unwrap()
}

#[proc_macro]
pub fn number_of_tokens_in_a_prefixed_char_literal(_: TokenStream) -> TokenStream {
    TokenStream::from_str("hey#'a'").unwrap().into_iter().count().to_string().parse().unwrap()
}

#[proc_macro]
pub fn number_of_tokens_in_a_prefixed_string_literal(_: TokenStream) -> TokenStream {
    TokenStream::from_str("hey#\"abc\"").unwrap().into_iter().count().to_string().parse().unwrap()
}
