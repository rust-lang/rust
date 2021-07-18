// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![deny(warnings)]
#![feature(proc_macro_expand_literal, proc_macro_span)]

extern crate proc_macro;

use proc_macro::*;
use std::str::FromStr;

fn chomp_literal(iter: &mut token_stream::IntoIter) -> String {
    match iter.next() {
        Some(TokenTree::Punct(p)) => {
            assert_eq!(p.as_char(), '-');
            match iter.next() {
                Some(TokenTree::Literal(l)) => format!("- {}", l),
                _ => panic!("unexpected token after '-'"),
            }
        }
        Some(TokenTree::Literal(l)) => l.to_string(),
        Some(TokenTree::Group(g)) => {
            assert_eq!(g.delimiter(), Delimiter::None);
            let mut inner = g.stream().into_iter();
            let rv = chomp_literal(&mut inner);
            assert!(inner.next().is_none());
            rv
        }
        _ => panic!("expected literal"),
    }
}

#[proc_macro]
pub fn expand_literal_is(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let expected = chomp_literal(&mut iter);
    match iter.next() {
        Some(TokenTree::Punct(ref p)) if p.as_char() == ',' => {}
        _ => panic!("expected comma"),
    }

    let expanded = iter.collect::<TokenStream>().expand_literal().expect("expand_literal failed");
    assert!(
        expected == expanded.to_string(),
        "assert failed\nexpected: `{:?}`\nexpanded: `{:?}`",
        expected,
        expanded.to_string()
    );
    assert_eq!(expected, expanded.to_string());

    let span = expanded.span();
    assert!(span.eq(&span.resolved_at(Span::call_site())), "span discards context");

    TokenStream::new()
}

#[proc_macro]
pub fn recursive_expand(_: TokenStream) -> TokenStream {
    // Recursively call until we hit the recursion limit and get an error.
    //
    // NOTE: This doesn't panic if expansion fails because that'll cause a very
    // large number of errors to fill the output.
    let lit = TokenStream::from_str("recursive_expand!{}")
        .unwrap()
        .expand_literal()
        .unwrap_or(Literal::u32_suffixed(0));
    std::iter::once(TokenTree::Literal(lit)).collect()
}

#[proc_macro]
pub fn echo_pm(input: TokenStream) -> TokenStream {
    input
}
