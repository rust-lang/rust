extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn tester(_: TokenStream, _: TokenStream) -> TokenStream {
    panic!();
}
