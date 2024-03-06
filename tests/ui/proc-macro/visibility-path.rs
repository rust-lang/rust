// Proc macro defined with `pub(path)` doesn't ICEs due to resolving the `path` (issue #68921).

//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub(self) fn outer(input: TokenStream) -> TokenStream {
    //~^ ERROR functions tagged with `#[proc_macro]` must be `pub`
    input
}

mod m {
    use proc_macro::*;

    #[proc_macro]
    pub(super) fn inner(input: TokenStream) -> TokenStream {
        //~^ ERROR functions tagged with `#[proc_macro]` must currently reside in the root
        input
    }
}
