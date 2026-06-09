extern crate proc_macro;
use proc_macro::*;

#[proc_macro_attribute]
pub fn a(_: TokenStream, input: TokenStream) -> TokenStream {
   input
}
