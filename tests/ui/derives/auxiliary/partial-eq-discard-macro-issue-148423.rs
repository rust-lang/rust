extern crate proc_macro;

use proc_macro::TokenStream;

/// Discards the item this attribute is applied to.
#[proc_macro_attribute]
pub fn discard(_: TokenStream, _: TokenStream) -> TokenStream {
    TokenStream::new()
}
