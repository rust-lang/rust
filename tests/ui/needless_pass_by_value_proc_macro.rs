#![crate_type = "proc-macro"]
#![warn(clippy::needless_pass_by_value)]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Foo)]
pub fn foo(_input: TokenStream) -> TokenStream {
    unimplemented!()
}
