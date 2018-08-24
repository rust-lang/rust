// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Append)]
pub fn derive_a(input: TokenStream) -> TokenStream {
    "impl Append for A {
         fn foo(&self) {}
     }
    ".parse().unwrap()
}
