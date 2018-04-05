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

#![crate_type = "proc-macro"]
#![feature(proc_macro)]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn nop_attr(_attr: TokenStream, input: TokenStream) -> TokenStream {
    assert!(_attr.to_string().is_empty());
    input
}

#[proc_macro_attribute]
pub fn no_output(_attr: TokenStream, _input: TokenStream) -> TokenStream {
    assert!(_attr.to_string().is_empty());
    assert!(!_input.to_string().is_empty());
    "".parse().unwrap()
}

#[proc_macro]
pub fn emit_input(input: TokenStream) -> TokenStream {
    input
}
