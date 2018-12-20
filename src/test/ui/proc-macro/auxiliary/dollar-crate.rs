// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn m(input: TokenStream) -> TokenStream {
    println!("PROC MACRO INPUT (PRETTY-PRINTED): {}", input);
    println!("PROC MACRO INPUT: {:#?}", input);
    input.into_iter().collect()
}

#[proc_macro_attribute]
pub fn a(_args: TokenStream, input: TokenStream) -> TokenStream {
    println!("ATTRIBUTE INPUT (PRETTY-PRINTED): {}", input);
    println!("ATTRIBUTE INPUT: {:#?}", input);
    input.into_iter().collect()
}

#[proc_macro_derive(d)]
pub fn d(input: TokenStream) -> TokenStream {
    println!("DERIVE INPUT (PRETTY-PRINTED): {}", input);
    println!("DERIVE INPUT: {:#?}", input);
    input.into_iter().collect()
}
