// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

#[proc_macro_derive(A)]
pub fn derive_a(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    "fn f() { println!(\"{}\", foo); }".parse().unwrap()
}
