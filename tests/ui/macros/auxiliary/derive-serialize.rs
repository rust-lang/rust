extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Serialize)]
pub fn derive_serialize(_: TokenStream) -> TokenStream {
    TokenStream::new()
}
