// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(Derive)]
pub fn derive(_: TokenStream) -> TokenStream {
    let code = "
        fn one(r: Restricted) {
            r.field;
        }
        fn two(r: Restricted) {
            r.field;
        }
    ";

    code.parse().unwrap()
}
