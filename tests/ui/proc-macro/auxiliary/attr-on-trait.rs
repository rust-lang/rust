extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn foo(attr: TokenStream, item: TokenStream) -> TokenStream {
    drop(attr);
    assert_eq!(item.to_string(), "fn foo() {}");
    "fn foo(&self);".parse().unwrap()
}
