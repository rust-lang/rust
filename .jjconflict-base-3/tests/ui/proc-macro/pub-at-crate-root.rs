//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

pub mod a { //~ ERROR `proc-macro` crate types currently cannot export any items
    use proc_macro::TokenStream;

    #[proc_macro_derive(B)]
    pub fn bar(a: TokenStream) -> TokenStream {
    //~^ ERROR: must currently reside in the root of the crate
        a
    }
}

#[proc_macro_derive(B)]
fn bar(a: proc_macro::TokenStream) -> proc_macro::TokenStream {
//~^ ERROR: functions tagged with `#[proc_macro_derive]` must be `pub`
    a
}
