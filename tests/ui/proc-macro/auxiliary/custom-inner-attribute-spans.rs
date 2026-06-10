extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree};

#[proc_macro_attribute]
pub fn check_spans(_: TokenStream, item: TokenStream) -> TokenStream {
    let mut item_iterator = item.clone().into_iter();

    for _ in 0..4 {
        item_iterator.next().unwrap();
    }

    let span = item_iterator.next().unwrap().span();
    let file = span.file();

    assert!(file.contains("auxiliary/custom-inner-attribute-spans-module.rs"));
    item
}
