// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Foo)]
pub fn derive_foo(input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_derive(Bar)]
pub fn derive_bar(input: TokenStream) -> TokenStream {
    panic!("lolnope");
}

#[proc_macro_derive(WithHelper, attributes(helper))]
pub fn with_helper(input: TokenStream) -> TokenStream {
    TokenStream::new()
}

#[proc_macro_attribute]
pub fn helper(_: TokenStream, input: TokenStream) -> TokenStream {
    input
}
