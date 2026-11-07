#![crate_type = "proc-macro"]

extern crate proc_macro;

#[proc_macro_derive(A)]
pub fn derive(ts: proc_macro::TokenStream) -> proc_macro::TokenStream {
    ts
}

#[derive(Debug)]
struct S;
