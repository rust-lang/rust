// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "proc-macro"]
#![allow(warnings)]

extern crate proc_macro;

#[proc_macro_derive(A)]
pub unsafe extern fn foo(a: i32, b: u32) -> u32 {
    //~^ ERROR: mismatched types
    //~| NOTE: expected normal fn, found unsafe fn
    //~| NOTE: expected type `fn(proc_macro::TokenStream) -> proc_macro::TokenStream`
    loop {}
}
