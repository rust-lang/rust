extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(AToB)]
pub fn derive(input: TokenStream) -> TokenStream {
    let input = input.to_string();
    assert_eq!(input, "struct A;");
    "struct B;".parse().unwrap()
}
