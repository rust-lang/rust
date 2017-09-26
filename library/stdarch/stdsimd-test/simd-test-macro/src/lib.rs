//! Implementation of the `#[simd_test]` macro
//!
//! This macro expands to a `#[test]` function which tests the local machine for
//! the appropriate cfg before calling the inner test function.

#![feature(proc_macro)]

#[macro_use]
extern crate quote;
extern crate proc_macro;
extern crate proc_macro2;

use proc_macro2::{TokenStream, Term, TokenNode, TokenTree};
use proc_macro2::Literal;

fn string(s: &str) -> TokenTree {
    TokenTree {
        kind: TokenNode::Literal(Literal::string(s)),

        span: Default::default(),
    }
}

#[proc_macro_attribute]
pub fn simd_test(attr: proc_macro::TokenStream,
                 item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let tokens = TokenStream::from(attr).into_iter().collect::<Vec<_>>();
    if tokens.len() != 2 {
        panic!("expected #[simd_test = \"feature\"]");
    }
    match tokens[0].kind {
        TokenNode::Op('=', _) => {}
        _ => panic!("expected #[simd_test = \"feature\"]"),
    }
    let target_feature = &tokens[1];
    let enable_feature = match tokens[1].kind {
        TokenNode::Literal(ref l) => l.to_string(),
        _ => panic!("expected #[simd_test = \"feature\"]"),
    };
    let enable_feature = enable_feature.trim_left_matches('"')
                                       .trim_right_matches('"');
    let enable_feature = string(&format!("+{}", enable_feature));
    let item = TokenStream::from(item);
    let name = find_name(item.clone());

    let name: TokenStream = name.as_str().parse().unwrap();

    let ret: TokenStream = quote! {
        #[test]
        fn #name() {
            if cfg_feature_enabled!(#target_feature) {
                return #name();
            }

            #[target_feature = #enable_feature]
            #item
        }
    }.into();
    ret.into()
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
