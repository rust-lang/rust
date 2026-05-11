// First version of the proc-macro. Emits `pub const ANSWER: u32 = 1;`.

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(ChangingDerive)]
pub fn changing_derive(_input: TokenStream) -> TokenStream {
    "pub const ANSWER: u32 = 1;".parse().unwrap()
}
