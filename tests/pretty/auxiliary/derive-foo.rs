extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Foo, attributes(Bar))]
pub fn derive(input: TokenStream) -> TokenStream {
    "".parse().unwrap()
}
