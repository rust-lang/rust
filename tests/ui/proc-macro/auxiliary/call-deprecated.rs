extern crate proc_macro;
use proc_macro::*;

#[proc_macro_attribute]
#[deprecated(since = "1.0.0", note = "test")]
pub fn attr(_: TokenStream, input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_attribute]
#[deprecated(since = "1.0.0", note = "test")]
pub fn attr_remove(_: TokenStream, _: TokenStream) -> TokenStream {
    TokenStream::new()
}
