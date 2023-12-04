// Check that we suggest *both* possible signatures of derive proc macros, namely
//     fn(TokenStream) -> TokenStream
// and
//     fn(TokenStream, DeriveExpansionOptions) -> TokenStream
// provided libs feature `derive_const` is enabled.

// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(derive_const)]

extern crate proc_macro;

#[proc_macro_derive(Blah)]
pub fn bad_input() -> proc_macro::TokenStream {
    //~^ ERROR derive proc macro has incorrect signature
    Default::default()
}
