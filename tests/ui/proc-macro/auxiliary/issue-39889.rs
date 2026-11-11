extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(Issue39889)]
pub fn f(_input: TokenStream) -> TokenStream {
    let rules = r#"
        macro_rules! id {
            ($($tt:tt)*) => { $($tt)* };
        }
    "#;
    rules.parse().unwrap()
}
