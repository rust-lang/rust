extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(has_helper_attr, attributes(helper))]
pub fn has_helper_attr(
    item: TokenStream,
) -> TokenStream {
    TokenStream::new()
}
