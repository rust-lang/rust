// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn test(_args: TokenStream, _item: TokenStream) -> TokenStream {
    TokenStream::default()
}
