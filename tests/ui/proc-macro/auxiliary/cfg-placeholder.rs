extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn my_proc_macro(_: TokenStream, input: TokenStream) -> TokenStream {
    if format!("{input:#?}").contains("my_attr1") {
        panic!("found gated attribute my_attr1");
    }
    if !format!("{input:#?}").contains("my_attr2") {
        panic!("didn't if gated my_attr2");
    }
    input
}

#[proc_macro_attribute]
pub fn my_attr1(_: TokenStream, input: TokenStream) -> TokenStream {
    panic!("my_attr1 was called");
    input
}

#[proc_macro_attribute]
pub fn my_attr2(_: TokenStream, input: TokenStream) -> TokenStream {
    if format!("{input:#?}").contains("my_attr1") {
        panic!("found gated attribute my_attr1");
    }
    input
}
