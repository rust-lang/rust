extern crate proc_macro;

use proc_macro::*;

#[proc_macro_derive(A, attributes(b))]
pub fn foo(_x: TokenStream) -> TokenStream {
    TokenStream::new()
}
