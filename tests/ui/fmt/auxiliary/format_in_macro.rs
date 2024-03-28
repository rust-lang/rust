// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(derive)]
pub fn derive(input: TokenStream) -> TokenStream {
    r#"fn derive_fn(text: impl ::std::fmt::Display) {
    println!(text);
}"#.parse().unwrap()
}

#[proc_macro]
pub fn function(input: TokenStream) -> TokenStream {
    r#"fn function_fn(text: impl ::std::fmt::Display) {
    println!(text);
}"#.parse().unwrap()
}

#[proc_macro_attribute]
pub fn attribute(input: TokenStream, attribute: TokenStream) -> TokenStream {
    r#"fn attribute_fn(text: impl ::std::fmt::Display) {
    println!(text);
}"#.parse().unwrap()
}
