// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![deny(warnings)]
#![feature(proc_macro_expand, proc_macro_span)]

extern crate proc_macro;

use proc_macro::*;
use std::str::FromStr;

#[proc_macro]
pub fn expand_expr_is(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let mut expected_tts = Vec::new();
    loop {
        match iter.next() {
            Some(TokenTree::Punct(ref p)) if p.as_char() == ',' => break,
            Some(tt) => expected_tts.push(tt),
            None => panic!("expected comma"),
        }
    }

    let expected = expected_tts.into_iter().collect::<TokenStream>();
    let expanded = iter.collect::<TokenStream>().expand_expr().expect("expand_expr failed");
    assert!(
        expected.to_string() == expanded.to_string(),
        "assert failed\nexpected: `{:?}`\nexpanded: `{:?}`",
        expected.to_string(),
        expanded.to_string()
    );

    TokenStream::new()
}

#[proc_macro]
pub fn recursive_expand(_: TokenStream) -> TokenStream {
    // Recursively call until we hit the recursion limit and get an error.
    //
    // NOTE: This doesn't panic if expansion fails because that'll cause a very
    // large number of errors to fill the output.
    TokenStream::from_str("recursive_expand!{}")
        .unwrap()
        .expand_expr()
        .unwrap_or(std::iter::once(TokenTree::Literal(Literal::u32_suffixed(0))).collect())
}

#[proc_macro]
pub fn echo_pm(input: TokenStream) -> TokenStream {
    input
}
