extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(FooWithLongName)]
pub fn derive_foo(input: TokenStream) -> TokenStream {
    "".parse().unwrap()
}
