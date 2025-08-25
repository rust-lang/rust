extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn check(_: TokenStream) -> TokenStream {
    assert_eq!(6, "---\n---".parse::<TokenStream>().unwrap().into_iter().count());
    Default::default()
}
