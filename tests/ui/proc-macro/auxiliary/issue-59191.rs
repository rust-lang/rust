// edition:2018
// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn no_main(_attrs: TokenStream, _input: TokenStream) -> TokenStream {
    let new_krate = r#"
        fn main() {}
    "#;
    new_krate.parse().unwrap()
}
