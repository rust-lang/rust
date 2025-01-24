//@ edition: 2021

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
