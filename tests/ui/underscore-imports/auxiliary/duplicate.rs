// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro_attribute]
pub fn duplicate(_: TokenStream, input: TokenStream) -> TokenStream {
    let clone = input.clone();
    input.into_iter().chain(clone.into_iter()).collect()
}
