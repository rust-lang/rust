//! Implementation of the `#[simd_test]` macro
//!
//! This macro expands to a `#[test]` function which tests the local machine
//! for the appropriate cfg before calling the inner test function.
#![deny(rust_2018_idioms)]

#[macro_use]
extern crate quote;

use proc_macro2::{Delimiter, Ident, Literal, Span, TokenStream, TokenTree};
use quote::ToTokens;
use std::env;

fn string(s: &str) -> TokenTree {
    Literal::string(s).into()
}

#[proc_macro_attribute]
pub fn simd_test(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let tokens = TokenStream::from(attr).into_iter().collect::<Vec<_>>();
    if tokens.len() != 3 {
        panic!("expected #[simd_test(enable = \"feature\")]");
    }
    match &tokens[0] {
        TokenTree::Ident(tt) if *tt == "enable" => {}
        _ => panic!("expected #[simd_test(enable = \"feature\")]"),
    }
    match &tokens[1] {
        TokenTree::Punct(tt) if tt.as_char() == '=' => {}
        _ => panic!("expected #[simd_test(enable = \"feature\")]"),
    }
    let enable_feature = match &tokens[2] {
        TokenTree::Literal(tt) => tt.to_string(),
        _ => panic!("expected #[simd_test(enable = \"feature\")]"),
    };
    let enable_feature = enable_feature.trim_start_matches('"').trim_end_matches('"');
    let target_features: Vec<String> = enable_feature
        .replace('+', "")
        .split(',')
        .map(String::from)
        .collect();

    let enable_feature = string(enable_feature);
    let item = TokenStream::from(item);
    let name = find_name(item.clone());

    let name: TokenStream = name
        .to_string()
        .parse()
        .unwrap_or_else(|_| panic!("failed to parse name: {}", name.to_string()));

    let target = env::var("TARGET").expect(
        "TARGET environment variable should be set for rustc (e.g. TARGET=x86_64-apple-darwin cargo test)"
    );
    let mut force_test = false;
    let macro_test = match target
        .split('-')
        .next()
        .unwrap_or_else(|| panic!("target triple contained no \"-\": {}", target))
    {
        "i686" | "x86_64" | "i586" => "is_x86_feature_detected",
        "arm" | "armv7" => "is_arm_feature_detected",
        "aarch64" => "is_aarch64_feature_detected",
        "powerpc" | "powerpcle" => "is_powerpc_feature_detected",
        "powerpc64" | "powerpc64le" => "is_powerpc64_feature_detected",
        "mips" | "mipsel" | "mipsisa32r6" | "mipsisa32r6el" => {
            // FIXME:
            // On MIPS CI run-time feature detection always returns false due
            // to this qemu bug: https://bugs.launchpad.net/qemu/+bug/1754372
            //
            // This is a workaround to force the MIPS tests to always run on
            // CI.
            force_test = true;
            "is_mips_feature_detected"
        }
        "mips64" | "mips64el" | "mipsisa64r6" | "mipsisa64r6el" => {
            // FIXME: see above
            force_test = true;
            "is_mips64_feature_detected"
        }
        t => panic!("unknown target: {}", t),
    };
    let macro_test = Ident::new(macro_test, Span::call_site());

    let mut cfg_target_features = TokenStream::new();
    for feature in target_features {
        let q = quote_spanned! {
            proc_macro2::Span::call_site() =>
            #macro_test!(#feature) &&
        };
        q.to_tokens(&mut cfg_target_features);
    }
    let q = quote! { true };
    q.to_tokens(&mut cfg_target_features);

    let test_norun = std::env::var("STDSIMD_TEST_NORUN").is_ok();
    let maybe_ignore = if test_norun {
        quote! { #[ignore] }
    } else {
        TokenStream::new()
    };

    let ret: TokenStream = quote_spanned! {
        proc_macro2::Span::call_site() =>
        #[allow(non_snake_case)]
        #[test]
        #maybe_ignore
        fn #name() {
            if #force_test | (#cfg_target_features) {
                let v = unsafe { #name() };
                return v;
            } else {
                ::stdarch_test::assert_skip_test_ok(stringify!(#name));
            }

            #[target_feature(enable = #enable_feature)]
            #item
        }
    };
    ret.into()
}

fn find_name(item: TokenStream) -> Ident {
    let mut tokens = item.into_iter();
    while let Some(tok) = tokens.next() {
        if let TokenTree::Ident(word) = tok {
            if word == "fn" {
                break;
            }
        }
    }

    fn get_ident(tt: TokenTree) -> Option<Ident> {
        match tt {
            TokenTree::Ident(i) => Some(i),
            TokenTree::Group(g) if g.delimiter() == Delimiter::None => {
                get_ident(g.stream().into_iter().next()?)
            }
            _ => None,
        }
    }

    tokens
        .next()
        .and_then(get_ident)
        .expect("failed to find function name")
}
