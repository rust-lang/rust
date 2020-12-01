// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn dummy(input: TokenStream) -> TokenStream {
    // Iterate to force internal conversion of nonterminals
    // to `proc_macro` structs
    for _ in input {}
    TokenStream::new()
}
