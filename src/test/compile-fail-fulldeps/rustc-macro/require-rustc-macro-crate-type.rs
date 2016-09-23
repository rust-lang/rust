// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_macro)]

extern crate rustc_macro;

#[rustc_macro_derive(Foo)]
//~^ ERROR: only usable with crates of the `rustc-macro` crate type
pub fn foo(a: rustc_macro::TokenStream) -> rustc_macro::TokenStream {
    a
}

fn main() {}
