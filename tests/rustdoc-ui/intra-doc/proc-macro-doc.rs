// check-pass
// force-host
// no-prefer-dynamic
// compile-flags: --crate-type proc-macro

#![deny(rustdoc::broken_intra_doc_links)]

extern crate proc_macro;
use proc_macro::*;

/// [`Unpin`]
#[proc_macro_derive(F)]
pub fn derive_(t: proc_macro::TokenStream) -> proc_macro::TokenStream {
    t
}

/// [`Vec`]
#[proc_macro_attribute]
pub fn attr(t: proc_macro::TokenStream, _: proc_macro::TokenStream) -> proc_macro::TokenStream {
    t
}

/// [`std::fs::File`]
#[proc_macro]
pub fn func(t: proc_macro::TokenStream) -> proc_macro::TokenStream {
    t
}
