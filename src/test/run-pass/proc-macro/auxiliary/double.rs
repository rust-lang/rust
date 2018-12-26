// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

// Outputs another copy of the struct.  Useful for testing the tokens
// seen by the proc_macro.
#[proc_macro_derive(Double)]
pub fn derive(input: TokenStream) -> TokenStream {
    format!("mod foo {{ {} }}", input.to_string()).parse().unwrap()
}
