//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro_derive(Serialize, attributes(serde))]
pub fn serialize(ts: TokenStream) -> TokenStream {
    quote!{}
}

#[proc_macro_derive(Deserialize, attributes(serde))]
pub fn deserialize(ts: TokenStream) -> TokenStream {
    quote!{}
}
