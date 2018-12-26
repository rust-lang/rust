// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

fn foo(arg: TokenStream) -> TokenStream {
    #[proc_macro]
    pub fn foo(arg: TokenStream) -> TokenStream { arg }
    //~^ ERROR functions tagged with `#[proc_macro]` must currently reside in the root of the crate

    arg
}
