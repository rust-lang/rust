// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
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
#![feature(proc_macro)]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Test)]
pub fn derive(_input: TokenStream) -> TokenStream {
    "fn f(s: S) { s.x }".parse().unwrap()
}

#[proc_macro_attribute]
pub fn attr_test(_attr: TokenStream, input: TokenStream) -> TokenStream {
    input
}
