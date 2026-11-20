#![allow(text_direction_codepoint_in_literal)]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn create_rtl_in_string(_: TokenStream) -> TokenStream {
    r#""‮test⁦ RTL in string literal""#.parse().unwrap()
}

#[proc_macro]
pub fn forward_stream(s: TokenStream) -> TokenStream {
    s
}

#[proc_macro]
pub fn recollect_stream(s: TokenStream) -> TokenStream {
    s.into_iter().collect()
}
