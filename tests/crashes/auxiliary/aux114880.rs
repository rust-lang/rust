#![feature(proc_macro_expand)]
#![crate_type = "proc-macro"]

extern crate proc_macro;
use std::str::FromStr;

use proc_macro::TokenStream;

#[proc_macro]
pub fn expand(_: TokenStream) -> TokenStream {
    dbg!(TokenStream::from_str("include!(\"./doesnt_exist\")").unwrap().expand_expr())
        .unwrap_or_default()
}
