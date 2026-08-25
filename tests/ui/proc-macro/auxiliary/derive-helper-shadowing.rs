//@revisions: edition2015 edition2018
//@[edition2015] edition:2015
//@[edition2018] edition:2018
extern crate proc_macro;
use proc_macro::*;

#[proc_macro_derive(GenHelperUse)]
pub fn derive_a(_: TokenStream) -> TokenStream {
    "
    #[empty_helper]
    struct Uwu;
    ".parse().unwrap()
}
