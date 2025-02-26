extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Foo, attributes(foo))]
pub fn derive(input: TokenStream) -> TokenStream {
    assert!(!input.to_string().contains("#[cfg(any())]"));
    "".parse().unwrap()
}
