// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(PartialEq)]
//~^ ERROR: cannot override a built-in #[derive] mode
pub fn foo(input: TokenStream) -> TokenStream {
    input
}
