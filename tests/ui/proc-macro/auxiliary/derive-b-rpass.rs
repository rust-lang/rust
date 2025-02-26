extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(B, attributes(B, C))]
pub fn derive(input: TokenStream) -> TokenStream {
    let input = input.to_string();
    assert!(input.contains("#[B[arbitrary tokens]]"));
    assert!(input.contains("struct B {"));
    assert!(input.contains("#[C]"));
    "".parse().unwrap()
}
