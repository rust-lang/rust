// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![deny(warnings)]
#![feature(proc_macro_expand, proc_macro_span)]

extern crate proc_macro;

use proc_macro::*;
use std::str::FromStr;

// Flatten the TokenStream, removing any toplevel `Delimiter::None`s for
// comparison.
fn flatten(ts: TokenStream) -> Vec<TokenTree> {
    ts.into_iter()
        .flat_map(|tt| match &tt {
            TokenTree::Group(group) if group.delimiter() == Delimiter::None => {
                flatten(group.stream())
            }
            _ => vec![tt],
        })
        .collect()
}

// Assert that two TokenStream values are roughly equal to one-another.
fn assert_ts_eq(lhs: &TokenStream, rhs: &TokenStream) {
    let ltts = flatten(lhs.clone());
    let rtts = flatten(rhs.clone());

    if ltts.len() != rtts.len() {
        panic!(
            "expected the same number of tts ({} == {})\nlhs:\n{:#?}\nrhs:\n{:#?}",
            ltts.len(),
            rtts.len(),
            lhs,
            rhs
        )
    }

    for (ltt, rtt) in ltts.iter().zip(&rtts) {
        match (ltt, rtt) {
            (TokenTree::Group(l), TokenTree::Group(r)) => {
                assert_eq!(
                    l.delimiter(),
                    r.delimiter(),
                    "expected delimiters to match for {:?} and {:?}",
                    l,
                    r
                );
                assert_ts_eq(&l.stream(), &r.stream());
            }
            (TokenTree::Punct(l), TokenTree::Punct(r)) => assert_eq!(
                (l.as_char(), l.spacing()),
                (r.as_char(), r.spacing()),
                "expected punct to match for {:?} and {:?}",
                l,
                r
            ),
            (TokenTree::Ident(l), TokenTree::Ident(r)) => assert_eq!(
                l.to_string(),
                r.to_string(),
                "expected ident to match for {:?} and {:?}",
                l,
                r
            ),
            (TokenTree::Literal(l), TokenTree::Literal(r)) => assert_eq!(
                l.to_string(),
                r.to_string(),
                "expected literal to match for {:?} and {:?}",
                l,
                r
            ),
            (l, r) => panic!("expected type to match for {:?} and {:?}", l, r),
        }
    }
}

#[proc_macro]
pub fn expand_expr_is(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let mut expected_tts = Vec::new();
    let comma = loop {
        match iter.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ',' => break p,
            Some(tt) => expected_tts.push(tt),
            None => panic!("expected comma"),
        }
    };

    // Make sure that `Ident` and `Literal` objects from this proc-macro's
    // environment are not invalidated when `expand_expr` recursively invokes
    // another macro by taking a local copy, and checking it after the fact.
    let pre_expand_span = comma.span();
    let pre_expand_ident = Ident::new("ident", comma.span());
    let pre_expand_literal = Literal::string("literal");
    let pre_expand_call_site = Span::call_site();

    let expected = expected_tts.into_iter().collect::<TokenStream>();
    let expanded = iter.collect::<TokenStream>().expand_expr().expect("expand_expr failed");
    assert!(
        expected.to_string() == expanded.to_string(),
        "assert failed\nexpected: `{}`\nexpanded: `{}`",
        expected.to_string(),
        expanded.to_string()
    );

    // Also compare the raw tts to make sure they line up.
    assert_ts_eq(&expected, &expanded);

    assert!(comma.span().eq(&pre_expand_span), "pre-expansion span is still equal");
    assert_eq!(pre_expand_ident.to_string(), "ident", "pre-expansion identifier is still valid");
    assert_eq!(
        pre_expand_literal.to_string(),
        "\"literal\"",
        "pre-expansion literal is still valid"
    );
    assert!(Span::call_site().eq(&pre_expand_call_site), "pre-expansion call-site is still equal");

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
