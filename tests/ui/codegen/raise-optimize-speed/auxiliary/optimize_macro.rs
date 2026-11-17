//@ compile-flags: -Copt-level=0 -Zraise-optimize-speed=modules,probe=added

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn optimize(_args: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn used(_args: TokenStream, item: TokenStream) -> TokenStream {
    item
}
