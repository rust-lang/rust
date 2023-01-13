// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn vec_ice(_attr: TokenStream, input: TokenStream) -> TokenStream {
    // This redundant convert is necessary to reproduce ICE.
    input.into_iter().collect()
}
