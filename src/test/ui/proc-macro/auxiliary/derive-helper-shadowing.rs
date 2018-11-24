// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro_attribute]
pub fn my_attr(_: TokenStream, input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_derive(MyTrait, attributes(my_attr))]
pub fn derive(input: TokenStream) -> TokenStream {
    TokenStream::new()
}
