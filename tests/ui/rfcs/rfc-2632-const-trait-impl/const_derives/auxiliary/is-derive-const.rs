// force-host
// no-prefer-dynamic
#![crate_type = "proc-macro"]
#![feature(derive_const)]

extern crate proc_macro;

use proc_macro::{TokenStream, DeriveExpansionOptions};

#[proc_macro_derive(IsDeriveConst)]
pub fn is_derive_const(_: TokenStream, options: DeriveExpansionOptions) -> TokenStream {
    format!("const IS_DERIVE_CONST: bool = {};", options.is_const()).parse().unwrap()
}
