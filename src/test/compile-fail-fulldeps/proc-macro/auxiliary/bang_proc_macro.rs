// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn bang_proc_macro(input: TokenStream) -> TokenStream {
    input
}
