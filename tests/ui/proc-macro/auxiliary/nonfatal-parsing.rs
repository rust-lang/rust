//@ edition: 2024
extern crate proc_macro;
use proc_macro::*;

#[path = "nonfatal-parsing-body.rs"]
mod body;

#[proc_macro]
pub fn run(_: TokenStream) -> TokenStream {
    body::run();
    TokenStream::new()
}
