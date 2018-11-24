// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro_derive(MyTrait, attributes(my_attr))]
pub fn foo(_: TokenStream) -> TokenStream {
    TokenStream::new()
}
