extern crate proc_macro;
use proc_macro::*;

#[proc_macro_derive(same_name, attributes(same_name))]
pub fn derive_a(_: TokenStream) -> TokenStream {
    TokenStream::new()
}
