//@ edition: 2018

extern crate proc_macro;

use proc_macro::TokenStream;
use std::str::FromStr;

#[proc_macro]
pub fn number_of_tokens(_: TokenStream) -> TokenStream {
    TokenStream::from_str("c\"\"").unwrap().into_iter().count().to_string().parse().unwrap()
}
