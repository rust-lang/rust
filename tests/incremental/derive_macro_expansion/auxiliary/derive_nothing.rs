//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(Nothing)]
pub fn derive(input: TokenStream) -> TokenStream {
    eprintln!("invoked");

    r#"
        pub mod nothing_mod {
            // #[cfg(cfail1)]
            pub fn nothing() {
                eprintln!("nothing");
            }

            // #[cfg(cfail2)]
            // fn nothingx() {}
        }
    "#.parse().unwrap()
}
