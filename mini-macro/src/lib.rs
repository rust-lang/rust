#![feature(proc_macro_quote)]
#![deny(rust_2018_idioms)]
// FIXME: Remove this attribute once the weird failure is gone.
#![allow(unused_extern_crates)]
extern crate proc_macro;

use proc_macro::{quote, TokenStream};

#[proc_macro_derive(ClippyMiniMacroTest)]
/// # Panics
///
/// Panics if the macro derivation fails
pub fn mini_macro(_: TokenStream) -> TokenStream {
    quote!(
        #[allow(unused)]
        fn needless_take_by_value(s: String) {
            println!("{}", s.len());
        }
        #[allow(unused)]
        fn needless_loop(items: &[u8]) {
            for i in 0..items.len() {
                println!("{}", items[i]);
            }
        }
        fn line_wrapper() {
            println!("{}", line!());
        }
    )
}
