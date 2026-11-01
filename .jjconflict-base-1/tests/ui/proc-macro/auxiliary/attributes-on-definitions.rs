#![feature(allow_internal_unsafe)]
#![feature(allow_internal_unstable)]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
#[allow_internal_unstable(proc_macro_internals)]
#[allow_internal_unsafe]
#[deprecated(since = "1.0.0", note = "test")]
pub fn with_attrs(_: TokenStream) -> TokenStream {
    "
    extern crate proc_macro;
    use ::proc_macro::bridge;

    fn contains_unsafe() { unsafe {} }
    ".parse().unwrap()
}
