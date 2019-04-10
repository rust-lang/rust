// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(repr128, proc_macro_hygiene, proc_macro_quote)]

extern crate proc_macro;

use proc_macro::{quote, TokenStream};

#[proc_macro_derive(DeriveSomething)]
pub fn derive(_: TokenStream) -> TokenStream {
    let output = quote! {
        #[allow(dead_code)]
        extern crate clippy_lints;
    };
    output
}
