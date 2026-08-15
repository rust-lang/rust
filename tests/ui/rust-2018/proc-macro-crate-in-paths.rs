//@ build-pass (FIXME(62277): could be check-pass?)
//@ force-host
//@ no-prefer-dynamic
//@ needs-unwind compiling proc macros with panic=abort causes a warning

#![crate_type = "proc-macro"]
#![deny(rust_2018_compatibility)]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Template, attributes(template))]
pub fn derive_template(input: TokenStream) -> TokenStream {
    input
}
