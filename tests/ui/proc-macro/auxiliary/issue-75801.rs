extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn foo(_args: TokenStream, item: TokenStream) -> TokenStream {
    item
}
