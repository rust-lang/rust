// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(AddImpl)]
// #[cfg(proc_macro)]
pub fn derive(input: TokenStream) -> TokenStream {
    "impl B {
            fn foo(&self) {}
        }

        fn foo() {}

        mod bar { pub fn foo() {} }
    ".parse().unwrap()
}
