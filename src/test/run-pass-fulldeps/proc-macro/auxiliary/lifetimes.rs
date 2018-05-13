// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic

#![feature(proc_macro)]
#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro]
pub fn lifetimes_bang(input: TokenStream) -> TokenStream {
    // Roundtrip through token trees
    input.into_iter().collect()
}

#[proc_macro_attribute]
pub fn lifetimes_attr(_: TokenStream, input: TokenStream) -> TokenStream {
    // Roundtrip through AST
    input
}

#[proc_macro_derive(Lifetimes)]
pub fn lifetimes_derive(input: TokenStream) -> TokenStream {
    // Roundtrip through a string
    format!("mod m {{ {} }}", input).parse().unwrap()
}
