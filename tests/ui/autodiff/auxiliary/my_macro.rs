//@ force-host
//@ no-prefer-dynamic
#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
#[macro_use]
pub fn autodiff(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item // identity proc-macro
}
