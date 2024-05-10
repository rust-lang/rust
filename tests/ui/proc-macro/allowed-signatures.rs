//@ check-pass
//@ force-host
//@ no-prefer-dynamic
//@ needs-unwind compiling proc macros with panic=abort causes a warning

#![crate_type = "proc-macro"]
#![allow(private_interfaces)]
extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn foo<T>(t: T) -> TokenStream {
    TokenStream::new()
}

trait Project {
    type Assoc;
}

impl Project for () {
    type Assoc = TokenStream;
}

#[proc_macro]
pub fn uwu(_input: <() as Project>::Assoc) -> <() as Project>::Assoc {
    TokenStream::new()
}
