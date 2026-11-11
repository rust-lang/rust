//@ force-host
//@ no-prefer-dynamic
//@ compile-flags: --crate-type proc-macro
#![crate_type="proc-macro"]
#![crate_name="proc_macro_inner"]

extern crate proc_macro;

use proc_macro::TokenStream;

/// Links to [`OtherDerive`]
#[proc_macro_derive(DeriveA)]
pub fn a_derive(input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_derive(OtherDerive)]
pub fn other_derive(input: TokenStream) -> TokenStream {
    input
}
