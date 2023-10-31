extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(DeriveSomething)]
pub fn derive(_: TokenStream) -> TokenStream {
    "fn _f() -> Vec<u8> { vec![] }".parse().unwrap()
}

#[proc_macro]
pub fn foo_bar(_: TokenStream) -> TokenStream {
    "fn issue_7422() { eprintln!(); }".parse().unwrap()
}
