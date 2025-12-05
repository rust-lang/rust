extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn id(_: TokenStream, input: TokenStream) -> TokenStream { input }
