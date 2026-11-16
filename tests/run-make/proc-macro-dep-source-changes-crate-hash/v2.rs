// Second version of the proc-macro. Source has changed: it now emits
// `pub const ANSWER: u32 = 2;`. Crate name and exported macro name match v1.

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(ChangingDerive)]
pub fn changing_derive(_input: TokenStream) -> TokenStream {
    "pub const ANSWER: u32 = 2;".parse().unwrap()
}
