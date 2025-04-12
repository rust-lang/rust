extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn check(_: TokenStream) -> TokenStream {
    assert!("---\n---".parse::<TokenStream>().unwrap().is_empty());
    Default::default()
}
