extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn drop_first_token(attr: TokenStream, input: TokenStream) -> TokenStream {
    assert!(attr.is_empty());
    input.into_iter().skip(1).collect()
}
