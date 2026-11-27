//@ edition: 2018

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn missing_unsafe(_input: TokenStream) -> TokenStream {
    "#[no_mangle] pub fn abc() {}".parse().unwrap()
}

#[proc_macro]
pub fn macro_rules_missing_unsafe(_input: TokenStream) -> TokenStream {
    "macro_rules! make_fn {
        () => { #[no_mangle] pub fn foo() { } };
    }"
    .parse()
    .unwrap()
}
