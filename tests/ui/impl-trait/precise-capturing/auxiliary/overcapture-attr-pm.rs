extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn rpc(_attr: TokenStream, _item: TokenStream) -> TokenStream {
    "pub fn generated(x: &u8) -> impl Sized { *x }".parse().unwrap()
}
