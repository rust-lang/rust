extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree};

#[proc_macro_attribute]
pub fn check_spans(_: TokenStream, item: TokenStream) -> TokenStream {
    item
    //let token_tree = item.into_iter().nth(4).unwrap();
    //let span = token_tree.span();
    //panic!("{} in {} for {}", span.line(), span.file(), token_tree)

    // panic!("{}", item);
    // let mut item_iterator = item.into_iter();

    // for _ in 0..4 {
    //     item_iterator.next();
    // }

    // let TokenTree::Group(group) = item_iterator.next().unwrap() else {
    //     panic!("These should be the module braces.");
    // };

    // let mut group = group.stream().into_iter();
    // //group.next();
    // let token_tree = group.next().expect("Token tree.");
    // let span = token_tree.span();
    // let file = span.file();

    // panic!("{} in {}", span.line(), file);

    // assert!(file.contains("custom-inner-attribute-spans-module.rs"));
    // //"mod tester { #![custom_inner_attribute_spans::check_spans] }".parse().unwrap()
    // TokenStream::new()
}
