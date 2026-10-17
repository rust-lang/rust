#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro_attribute]
pub fn attr_tru(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let name = item.into_iter().nth(1).unwrap();
    quote!(fn $name() -> bool { true })
}

#[proc_macro_attribute]
pub fn attr_identity(_attr: TokenStream, item: TokenStream) -> TokenStream {
    quote!($item)
}

#[proc_macro]
pub fn tru(_ts: TokenStream) -> TokenStream {
    quote!(true)
}

#[proc_macro]
pub fn ret_tru(_ts: TokenStream) -> TokenStream {
    quote!(return true;)
}

#[proc_macro]
pub fn identity(ts: TokenStream) -> TokenStream {
    quote!($ts)
}
