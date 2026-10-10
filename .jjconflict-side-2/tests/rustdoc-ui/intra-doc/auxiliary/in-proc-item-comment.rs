//@ force-host
//@ no-prefer-dynamic
#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

mod view {}

/// [`view`]
#[proc_macro]
pub fn f(_: TokenStream) -> TokenStream {
    todo!()
}

/// [`f()`]
#[proc_macro]
pub fn g(_: TokenStream) -> TokenStream {
    todo!()
}
