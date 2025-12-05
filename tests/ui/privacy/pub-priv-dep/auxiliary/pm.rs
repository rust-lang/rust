//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn fn_like(input: TokenStream) -> TokenStream {
    "".parse().unwrap()
}

#[proc_macro_derive(PmDerive)]
pub fn pm_derive(item: TokenStream) -> TokenStream {
    "".parse().unwrap()
}

#[proc_macro_attribute]
pub fn pm_attr(attr: TokenStream, item: TokenStream) -> TokenStream {
    "".parse().unwrap()
}
