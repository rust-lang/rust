//@ force-host
//@ no-prefer-dynamic
//@ edition: 2021

#![crate_type = "proc-macro"]
#![crate_name = "proc_macro_api_tests"]
#![feature(proc_macro_span)]
#![deny(dead_code)] // catch if a test function is never called

extern crate proc_macro;

mod cmp;
mod literal;

use proc_macro::TokenStream;

#[proc_macro]
pub fn run(input: TokenStream) -> TokenStream {
    assert!(input.is_empty());

    cmp::test();
    literal::test();

    TokenStream::new()
}
