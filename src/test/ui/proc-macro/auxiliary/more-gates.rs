// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro_attribute]
pub fn attr2mac1(_: TokenStream, _: TokenStream) -> TokenStream {
    "macro_rules! foo1 { (a) => (a) }".parse().unwrap()
}

#[proc_macro_attribute]
pub fn attr2mac2(_: TokenStream, _: TokenStream) -> TokenStream {
    "macro foo2(a) { a }".parse().unwrap()
}

#[proc_macro]
pub fn mac2mac1(_: TokenStream) -> TokenStream {
    "macro_rules! foo3 { (a) => (a) }".parse().unwrap()
}

#[proc_macro]
pub fn mac2mac2(_: TokenStream) -> TokenStream {
    "macro foo4(a) { a }".parse().unwrap()
}

#[proc_macro]
pub fn tricky(_: TokenStream) -> TokenStream {
    "fn foo() {
        macro_rules! foo { (a) => (a) }
    }".parse().unwrap()
}
