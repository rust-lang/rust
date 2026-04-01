extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree};

#[proc_macro]
pub fn print_tokens(input: TokenStream) -> TokenStream {
    println!("{:#?}", input);
    for token in input {
        println!("{token}");
    }
    TokenStream::new()
}
