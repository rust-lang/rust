//@ no-prefer-dynamic

#![crate_type = "proc-macro"]

use std::str::FromStr;

extern crate proc_macro;

#[proc_macro_derive(MyDeriveMacro)]
pub fn derive_my_derive_macro(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::from_str("
        #[macro_export]
        macro_rules! my_generated_macro {
            ($my_macro_parameter: expr) => {};
        }
    ").unwrap()
}
