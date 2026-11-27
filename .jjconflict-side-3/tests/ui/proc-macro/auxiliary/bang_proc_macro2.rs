extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn bang_proc_macro2(_: TokenStream) -> TokenStream {
    "let x = foobar2;".parse().unwrap()
}
