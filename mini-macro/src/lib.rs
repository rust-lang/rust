#![feature(use_extern_macros, proc_macro_quote, proc_macro_non_items)]
extern crate proc_macro;

use proc_macro::{TokenStream, quote};

#[proc_macro_derive(ClippyMiniMacroTest)]
pub fn mini_macro(_: TokenStream) -> TokenStream {
    quote!(
        #[allow(unused)] fn needless_take_by_value(s: String) { println!("{}", s.len()); }
    )
}