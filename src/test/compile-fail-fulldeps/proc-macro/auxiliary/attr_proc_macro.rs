// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn attr_proc_macro(_: TokenStream, input: TokenStream) -> TokenStream {
    input
}
