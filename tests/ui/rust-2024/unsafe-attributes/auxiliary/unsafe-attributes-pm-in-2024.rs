//@ edition: 2024

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn missing_unsafe(_input: TokenStream) -> TokenStream {
    "#[no_mangle] pub fn abc() {}".parse().unwrap()
}

#[proc_macro_attribute]
pub fn attr_missing_unsafe(_attr: TokenStream, _input: TokenStream) -> TokenStream {
    "#[no_mangle] pub fn bar() {}".parse().unwrap()
}

#[proc_macro_derive(AttrMissingUnsafe)]
pub fn derive_attr_missing_unsafe(_input: TokenStream) -> TokenStream {
    "#[no_mangle] pub fn baz() {}".parse().unwrap()
}

#[proc_macro]
pub fn macro_rules_missing_unsafe(_input: TokenStream) -> TokenStream {
    "macro_rules! make_fn {
        () => { #[no_mangle] pub fn foo() { } };
    }"
    .parse()
    .unwrap()
}
