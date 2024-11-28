#![crate_name="redundant_semi_proc_macro"]
extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn should_preserve_spans(_attr: TokenStream, item: TokenStream) -> TokenStream {
    eprintln!("{:?}", item);
    item
}
