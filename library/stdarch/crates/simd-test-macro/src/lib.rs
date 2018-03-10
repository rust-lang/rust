//! Implementation of the `#[simd_test]` macro
//!
//! This macro expands to a `#[test]` function which tests the local machine
//! for the appropriate cfg before calling the inner test function.

#![feature(proc_macro)]

extern crate proc_macro;
extern crate proc_macro2;
#[macro_use]
extern crate quote;

use std::env;

use proc_macro2::{Term, TokenNode, TokenStream, TokenTree};
use proc_macro2::Literal;

fn string(s: &str) -> TokenTree {
    TokenNode::Literal(Literal::string(s)).into()
}

#[proc_macro_attribute]
pub fn simd_test(
    attr: proc_macro::TokenStream, item: proc_macro::TokenStream
) -> proc_macro::TokenStream {
    let tokens = TokenStream::from(attr).into_iter().collect::<Vec<_>>();
    if tokens.len() != 2 {
        panic!("expected #[simd_test = \"feature\"]");
    }
    match tokens[0].kind {
        TokenNode::Op('=', _) => {}
        _ => panic!("expected #[simd_test = \"feature\"]"),
    }
    let target_features = match tokens[1].kind {
        TokenNode::Literal(ref l) => l.to_string(),
        _ => panic!("expected #[simd_test = \"feature\"]"),
    };
    let target_features: Vec<String> = target_features
        .replace('"', "")
        .replace('+', "")
        .split(',')
        .map(|v| String::from(v))
        .collect();

    let enable_feature = match tokens[1].kind {
        TokenNode::Literal(ref l) => l.to_string(),
        _ => panic!("expected #[simd_test = \"feature\"]"),
    };
    let enable_feature = enable_feature
        .trim_left_matches('"')
        .trim_right_matches('"');
    let enable_feature = string(enable_feature);
    let item = TokenStream::from(item);
    let name = find_name(item.clone());

    let name: TokenStream = name.as_str()
        .parse()
        .expect(&format!("failed to parse name: {}", name.clone().as_str()));

    let default_target = if cfg!(target_os = "windows") {
        Some("x86_64-pc-windows-msvc")
    } else if cfg!(target_os = "linux") {
        Some("x86_64-unknown-linux-gnu")
    } else if cfg!(target_os = "macos") {
        Some("x86_64-apple-darwin")
    } else {
        None
    };

    let target = env::var("TARGET").unwrap_or_else(|_| {
        default_target.expect("TARGET environment variable not set and no default target known for the current target.").to_string()
    });
    let mut force_test = false;
    let macro_test = match target
        .split('-')
        .next()
        .expect(&format!("target triple contained no \"-\": {}", target))
    {
        "i686" | "x86_64" | "i586" => "is_x86_feature_detected",
        "arm" => "is_arm_feature_detected",
        "aarch64" => "is_aarch64_feature_detected",
        "powerpc64" => "is_powerpc64_feature_detected",
        "mips" | "mipsel" => {
            // FIXME:
            // On MIPS CI run-time feature detection always returns false due
            // to this qemu bug: https://bugs.launchpad.net/qemu/+bug/1754372
            //
            // This is a workaround to force the MIPS tests to always run on
            // CI.
            force_test = true;
            "is_mips_feature_detected"
        }
        "mips64" | "mips64el" => {
            // FIXME: see above
            force_test = true;
            "is_mips64_feature_detected"
        }
        t => panic!("unknown target: {}", t),
    };
    let macro_test = proc_macro2::Term::intern(macro_test);

    let mut cfg_target_features = quote::Tokens::new();
    use quote::ToTokens;
    for feature in target_features {
        let q = quote_spanned! {
            proc_macro2::Span::call_site() =>
            #macro_test!(#feature) &&
        };
        q.to_tokens(&mut cfg_target_features);
    }
    let q = quote!{ true };
    q.to_tokens(&mut cfg_target_features);

    let ret: TokenStream = quote_spanned! {
        proc_macro2::Span::call_site() =>
        #[allow(non_snake_case)]
        #[test]
        fn #name() {
            if #force_test | (#cfg_target_features) {
                return unsafe { #name() };
            } else {
                ::stdsimd_test::assert_skip_test_ok(stringify!(#name));
            }

            #[target_feature(enable = #enable_feature)]
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
                break;
            }
        }
    }

    match tokens.next().map(|t| t.kind) {
        Some(TokenNode::Term(word)) => word,
        _ => panic!("failed to find function name"),
    }
}
