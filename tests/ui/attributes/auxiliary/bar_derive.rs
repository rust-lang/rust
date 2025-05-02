extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Bar, attributes(arg))]
pub fn derive_bar(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    TokenStream::new()
}
