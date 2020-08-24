//! Exports a few trivial procedural macros for testing.

use proc_macro::TokenStream;

#[proc_macro]
pub fn function_like_macro(args: TokenStream) -> TokenStream {
    args
}

#[proc_macro_attribute]
pub fn attribute_macro(_args: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_derive(DummyTrait)]
pub fn derive_macro(_item: TokenStream) -> TokenStream {
    TokenStream::new()
}
