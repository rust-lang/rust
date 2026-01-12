//@ force-host
//@ no-prefer-dynamic
//@ compile-flags: --crate-type proc-macro

#![crate_type = "proc-macro"]
#![crate_name = "just_some_proc"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn repro(_args: TokenStream, _input: TokenStream) -> TokenStream {
    "struct Repro;".parse().unwrap()
}
