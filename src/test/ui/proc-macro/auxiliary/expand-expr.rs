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
        "assert failed\nexpected: `{}`\nexpanded: `{}`",
        expected.to_string(),
        expanded.to_string()
    );

    TokenStream::new()
}

#[proc_macro]
pub fn expand_expr_fail(input: TokenStream) -> TokenStream {
    match input.expand_expr() {
        Ok(ts) => panic!("expand_expr unexpectedly succeeded: `{}`", ts),
        Err(_) => TokenStream::new(),
    }
}

#[proc_macro]
pub fn check_expand_expr_file(ts: TokenStream) -> TokenStream {
    // Check that the passed in `file!()` invocation and a parsed `file!`
    // invocation expand to the same literal.
    let input_t = ts.expand_expr().expect("expand_expr failed on macro input").to_string();
    let parse_t = TokenStream::from_str("file!{}")
    .unwrap()
        .expand_expr()
        .expect("expand_expr failed on internal macro")
        .to_string();
    assert_eq!(input_t, parse_t);

    // Check that the literal matches `Span::call_site().source_file().path()`
    let expect_t =
        Literal::string(&Span::call_site().source_file().path().to_string_lossy()).to_string();
    assert_eq!(input_t, expect_t);

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
