//@ edition: 2024
use proc_macro::{TokenStream, TokenTree};

#[proc_macro_attribute]
pub fn inspect_stream(_attr: TokenStream, item: TokenStream) -> TokenStream {
    for _token in item.clone().into_iter() {

    }

    item
}
