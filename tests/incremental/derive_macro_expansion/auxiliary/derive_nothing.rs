//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(Nothing)]
pub fn derive(_input: TokenStream) -> TokenStream {
    eprintln!("invoked");

    return r#"
        pub mod nothing_mod {
            pub fn nothing() {
                eprintln!("nothing");
            }
        }
    "#.parse().unwrap();
}
