// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn nop_attr(_attr: TokenStream, input: TokenStream) -> TokenStream {
    assert!(_attr.to_string().is_empty());
    input
}

#[proc_macro_attribute]
pub fn no_output(_attr: TokenStream, _input: TokenStream) -> TokenStream {
    assert!(_attr.to_string().is_empty());
    assert!(!_input.to_string().is_empty());
    "".parse().unwrap()
}

#[proc_macro]
pub fn emit_input(input: TokenStream) -> TokenStream {
    input
}
