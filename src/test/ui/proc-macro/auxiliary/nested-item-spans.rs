// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro_attribute]
pub fn foo(_: TokenStream, item: TokenStream) -> TokenStream {
    item.into_iter().collect()
}
