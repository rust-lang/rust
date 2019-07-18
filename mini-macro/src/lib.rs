#![feature(proc_macro_quote, proc_macro_hygiene)]
#![deny(rust_2018_idioms)]
extern crate proc_macro;

use proc_macro::{quote, TokenStream};

#[proc_macro_derive(ClippyMiniMacroTest)]
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
    )
}
