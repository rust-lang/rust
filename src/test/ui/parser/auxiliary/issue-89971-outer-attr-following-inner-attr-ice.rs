// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(ICE)]
pub fn derive(_: TokenStream) -> TokenStream {
    r#"#[allow(missing_docs)] struct X { }"#.parse().unwrap()
}
