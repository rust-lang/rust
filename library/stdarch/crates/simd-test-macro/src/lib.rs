//! Implementation of the `#[simd_test]` macro
//!
//! This macro expands to a `#[test]` function which tests the local machine
//! for the appropriate cfg before calling the inner test function.
#![deny(rust_2018_idioms)]

#[macro_use]
extern crate quote;

use proc_macro2::{Ident, Literal, Span, TokenStream, TokenTree};
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
    let mut item = syn::parse_macro_input!(item as syn::ItemFn);
    let item_attrs = std::mem::take(&mut item.attrs);
    let name = &item.sig.ident;

    let target = env::var("TARGET").expect(
        "TARGET environment variable should be set for rustc (e.g. TARGET=x86_64-apple-darwin cargo test)"
    );
    let macro_test = match target
        .split('-')
        .next()
        .unwrap_or_else(|| panic!("target triple contained no \"-\": {target}"))
    {
        "i686" | "x86_64" | "i586" => "is_x86_feature_detected",
        "arm" | "armv7" => "is_arm_feature_detected",
        "aarch64" | "arm64ec" | "aarch64_be" => "is_aarch64_feature_detected",
        maybe_riscv if maybe_riscv.starts_with("riscv") => "is_riscv_feature_detected",
        "powerpc" | "powerpcle" => "is_powerpc_feature_detected",
        "powerpc64" | "powerpc64le" => "is_powerpc64_feature_detected",
        "loongarch64" => "is_loongarch_feature_detected",
        "s390x" => "is_s390x_feature_detected",
        t => panic!("unknown target: {t}"),
    };
    let macro_test = Ident::new(macro_test, Span::call_site());

    let skipped_functions = env::var("STDARCH_TEST_SKIP_FUNCTION").unwrap_or_default();
    let skipped_features = env::var("STDARCH_TEST_SKIP_FEATURE").unwrap_or_default();

    let mut name_str = &*name.to_string();
    if name_str.starts_with("test_") {
        name_str = &name_str[5..];
    }

    let skip_this = skipped_functions
        .split(',')
        .map(str::trim)
        .any(|s| s == name_str)
        || skipped_features
            .split(',')
            .map(str::trim)
            .any(|s| target_features.iter().any(|feature| s == feature));

    let mut detect_missing_features = TokenStream::new();
    for feature in target_features {
        let q = quote_spanned! {
            proc_macro2::Span::call_site() =>
            if !#macro_test!(#feature) {
                missing_features.push(#feature);
            }
        };
        q.to_tokens(&mut detect_missing_features);
    }

    let maybe_ignore = if skip_this {
        quote! { #[ignore] }
    } else {
        TokenStream::new()
    };

    let ret: TokenStream = quote_spanned! {
        proc_macro2::Span::call_site() =>
        #[allow(non_snake_case)]
        #[test]
        #maybe_ignore
        #(#item_attrs)*
        fn #name() {
            let mut missing_features = ::std::vec::Vec::new();
            #detect_missing_features
            if missing_features.is_empty() {
                let v = unsafe { #name() };
                return v;
            } else {
                ::stdarch_test::assert_skip_test_ok(stringify!(#name), &missing_features);
            }

            #[target_feature(enable = #enable_feature)]
            #item
        }
    };
    ret.into()
}
