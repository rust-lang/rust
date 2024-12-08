extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(repro)]
pub fn proc_macro_hack_expr(_input: TokenStream) -> TokenStream {
    "macro_rules! m {()=>{}}".parse().unwrap()
}
