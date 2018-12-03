// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro_attribute]
pub fn cfg(_: TokenStream, input: TokenStream) -> TokenStream {
    //~^ ERROR name `cfg` is reserved in macro namespace
    input
}

#[proc_macro_attribute]
pub fn cfg_attr(_: TokenStream, input: TokenStream) -> TokenStream {
    //~^ ERROR name `cfg_attr` is reserved in macro namespace
    input
}

#[proc_macro_attribute]
pub fn derive(_: TokenStream, input: TokenStream) -> TokenStream {
    //~^ ERROR name `derive` is reserved in macro namespace
    input
}
