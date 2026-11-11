#![crate_name = "macro_dump_debug"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn dump_debug(tokens: TokenStream) -> TokenStream {
    eprintln!("{:?}", tokens);
    eprintln!("{:#?}", tokens);
    TokenStream::new()
}
