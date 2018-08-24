// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_non_items)]

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
