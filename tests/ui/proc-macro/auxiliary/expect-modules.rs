/// Macros that assert that certain `mod` items are present in their input.

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn expect_mod_item(_attrs: TokenStream, item: TokenStream) -> TokenStream {
    let s = item.to_string();

    assert_eq!(s, "mod module;");

    item
}

#[proc_macro_attribute]
pub fn expect_mods_attr(_attrs: TokenStream, item: TokenStream) -> TokenStream {
    let s = item.to_string();

    assert_contains(&s, "mod attr_outlined;");
    assert_contains(&s, "mod attr_inline {}");

    item
}

#[proc_macro_derive(ExpectModsDerive)]
pub fn expect_mods_derive(item: TokenStream) -> TokenStream {
    let s = item.to_string();

    assert_contains(&s, "mod derive_outlined;");
    assert_contains(&s, "mod derive_inline {}");

    TokenStream::new()
}

#[track_caller]
fn assert_contains(s: &str, needle: &str) {
    if !s.contains(needle) {
        panic!("{needle:?} not found in:\n---\n{s}\n---\n");
    }
}
