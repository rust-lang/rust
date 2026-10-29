//! Implementation of the `#[simd_test]` macro
//!
//! This macro expands to a `#[test]` function which tests the local machine
//! for the appropriate cfg before calling the inner test function.

use proc_macro2::{Span, TokenStream, TokenTree};
use quote::{ToTokens, quote, quote_spanned};
use std::env;

#[proc_macro_attribute]
pub fn simd_test(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let tokens = TokenStream::from(attr).into_iter().collect::<Vec<_>>();

    let (target_features, target_feature_attr) = match &tokens[..] {
        [] => (Vec::new(), TokenStream::new()),
        [
            TokenTree::Ident(enable),
            TokenTree::Punct(equals),
            TokenTree::Literal(literal),
        ] if enable == "enable" && equals.as_char() == '=' => {
            let lit = literal.to_string();
            let enable_feature = lit.trim_start_matches('"').trim_end_matches('"');
            let target_features: Vec<_> = enable_feature
                .replace('+', "")
                .split(',')
                .map(String::from)
                .collect();
            let mut enable_feature_arm = enable_feature.to_string();

            // Allows using `#[simd_test(enable = "neon")]` on aarch64/armv7 shared tests.
            if target_features.iter().any(|feat| feat == "neon") {
                enable_feature_arm.push_str(",v7");
            }

            (
                target_features,
                quote! {
                    #[cfg_attr(target_arch = "arm", target_feature(enable = #enable_feature_arm))]
                    #[cfg_attr(not(target_arch = "arm"), target_feature(enable = #enable_feature))]
                },
            )
        }
        _ => panic!("expected #[simd_test(enable = \"feature\")] or #[simd_test]"),
    };

    let mut item = syn::parse_macro_input!(item as syn::ItemFn);
    let item_attrs = std::mem::take(&mut item.attrs);
    let name = &item.sig.ident;

    let mut name_str = &*name.to_string();
    if name_str.starts_with("test_") {
        name_str = &name_str[5..];
    }

    let skipped_functions = env::var("STDARCH_TEST_SKIP_FUNCTION").unwrap_or_default();
    let skipped_features = env::var("STDARCH_TEST_SKIP_FEATURE").unwrap_or_default();
    let skipped_function = skipped_functions
        .split(',')
        .map(str::trim)
        .any(|s| s == name_str);
    let skipped_feature = skipped_features
        .split(',')
        .map(str::trim)
        .any(|s| target_features.iter().any(|feature| s == feature));

    let mut detect_missing_features = TokenStream::new();
    for feature in target_features {
        let mut q = quote_spanned! { Span::call_site() =>
            // Default target feature check
            if !test_macro!(#feature) {
                missing_features.push(#feature);
            }
        };

        // For features that cannot be checked at runtime (currently fp16 on arm32), use the
        // static check instead.
        if feature == "fp16" {
            q = quote_spanned! { Span::call_site() =>
                cfg_select! {
                    target_arch = "arm" => {
                        if !cfg!(target_feature = #feature) {
                            missing_features.push(#feature);
                        }
                    }
                    _ => {
                        #q
                    }
                }
            }
        }

        q.to_tokens(&mut detect_missing_features);
    }

    let mut maybe_ignore = TokenStream::new();
    let mut const_test = TokenStream::new();
    let mut const_stability = TokenStream::new();

    if skipped_function || skipped_feature {
        maybe_ignore = quote! { #[ignore] }
    }

    if item.sig.constness.is_some() {
        const_test = quote! {
            const _: () = unsafe { #name() };
        };
        const_stability = quote! {
            #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
        };
    }

    let ret: TokenStream = quote_spanned! { Span::call_site() =>
        #[allow(non_snake_case)]
        #[test]
        #maybe_ignore
        #(#item_attrs)*
        fn #name() {
            cfg_select! {
                any(target_arch = "x86", target_arch = "x86_64") => {
                    use std::arch::is_x86_feature_detected as test_macro;
                }
                target_arch = "arm" => {
                    use std::arch::is_arm_feature_detected as test_macro;
                }
                any(target_arch = "aarch64", target_arch = "arm64ec") => {
                    use std::arch::is_aarch64_feature_detected as test_macro;
                }
                any(target_arch = "riscv32", target_arch = "riscv64") => {
                    use std::arch::is_riscv_feature_detected as test_macro;
                }
                any(target_arch = "powerpc", target_arch = "powerpc64") => {
                    use std::arch::is_powerpc_feature_detected as test_macro;
                }
                any(target_arch = "loongarch32", target_arch = "loongarch64") => {
                    use std::arch::is_loongarch_feature_detected as test_macro;
                }
                target_arch = "s390x" => {
                    use std::arch::is_s390x_feature_detected as test_macro;
                }
                any(target_arch = "mips", target_arch = "mips32r6") => {
                    use std::arch::is_mips_feature_detected as test_macro;
                }
                any(target_arch = "mips64", target_arch = "mips64r6") => {
                    use std::arch::is_mips64_feature_detected as test_macro;
                }
                _ => compile_error!("unknown target; need to update simd-test-macro"),
            }

            #const_test

            let mut missing_features = ::std::vec::Vec::new();
            #detect_missing_features
            if missing_features.is_empty() {
                let v = unsafe { #name() };
                return v;
            } else {
                ::stdarch_test::assert_skip_test_ok(stringify!(#name), &missing_features);
            }

            #target_feature_attr
            #const_stability
            #item
        }
    };
    ret.into()
}
