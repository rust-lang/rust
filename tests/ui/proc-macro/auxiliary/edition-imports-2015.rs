//@ edition:2015

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Derive2015)]
pub fn derive_2015(_: TokenStream) -> TokenStream {
    "
    use import::Path;

    fn check_absolute() {
        let x = ::absolute::Path;
    }
    ".parse().unwrap()
}
