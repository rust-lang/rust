extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree};

#[proc_macro_attribute]
pub fn check_spans(_: TokenStream, item: TokenStream) -> TokenStream {
    let mut item = item.into_iter();

    for _ in 0..4 {
        item.next().unwrap();
    }

    let span = item.next().unwrap().span();
    let file = span.file();

    assert!(file.contains("auxiliary/custom-inner-attribute-spans-module.rs"));
    TokenStream::new()
}
