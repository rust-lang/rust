//! Implementation of the `#[assert_instr]` macro
//!
//! This macro is used when testing the `stdsimd` crate and is used to generate
//! test cases to assert that functions do indeed contain the instructions that
//! we're expecting them to contain.
//!
//! The procedural macro here is relatively simple, it simply appends a
//! `#[test]` function to the original token stream which asserts that the
//! function itself contains the relevant instruction.

#![feature(proc_macro)]

extern crate proc_macro;

use proc_macro::{TokenStream, Term, TokenNode, Delimiter};

#[proc_macro_attribute]
pub fn assert_instr(attr: TokenStream, item: TokenStream) -> TokenStream {
    let name = find_name(item.clone());
    let tokens = attr.into_iter().collect::<Vec<_>>();
    if tokens.len() != 1 {
        panic!("expected #[assert_instr(foo)]");
    }
    let tokens = match tokens[0].kind {
        TokenNode::Group(Delimiter::Parenthesis, ref rest) => rest.clone(),
        _ => panic!("expected #[assert_instr(foo)]"),
    };
    let tokens = tokens.into_iter().collect::<Vec<_>>();
    if tokens.len() != 1 {
        panic!("expected #[assert_instr(foo)]");
    }
    let instr = match tokens[0].kind {
        TokenNode::Term(term) => term,
        _ => panic!("expected #[assert_instr(foo)]"),
    };

    let ignore = if cfg!(optimized) {
        ""
    } else {
        "#[ignore]"
    };
    let test = format!("
        #[test]
        #[allow(non_snake_case)]
        {ignore}
        fn assert_instr_{name}() {{
            ::stdsimd_test::assert({name} as usize,
                                   \"{name}\",
                                   \"{instr}\");
        }}
    ", name = name.as_str(), instr = instr.as_str(), ignore = ignore);
    let test: TokenStream = test.parse().unwrap();

    item.into_iter().chain(test.into_iter()).collect()
}

fn find_name(item: TokenStream) -> Term {
    let mut tokens = item.into_iter();
    while let Some(tok) = tokens.next() {
        if let TokenNode::Term(word) = tok.kind {
            if word.as_str() == "fn" {
                break
            }
        }
    }

    match tokens.next().map(|t| t.kind) {
        Some(TokenNode::Term(word)) => word,
        _ => panic!("failed to find function name"),
    }
}
