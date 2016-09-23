// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "rustc-macro"]
#![feature(rustc_macro)]

extern crate rustc_macro;

#[rustc_macro_derive]
//~^ ERROR: attribute must be of form: #[rustc_macro_derive(TraitName)]
pub fn foo1(input: rustc_macro::TokenStream) -> rustc_macro::TokenStream {
    input
}

#[rustc_macro_derive = "foo"]
//~^ ERROR: attribute must be of form: #[rustc_macro_derive(TraitName)]
pub fn foo2(input: rustc_macro::TokenStream) -> rustc_macro::TokenStream {
    input
}

#[rustc_macro_derive(
    a = "b"
)]
//~^^ ERROR: must only be one word
pub fn foo3(input: rustc_macro::TokenStream) -> rustc_macro::TokenStream {
    input
}

#[rustc_macro_derive(b, c)]
//~^ ERROR: attribute must only have one argument
pub fn foo4(input: rustc_macro::TokenStream) -> rustc_macro::TokenStream {
    input
}

#[rustc_macro_derive(d(e))]
//~^ ERROR: must only be one word
pub fn foo5(input: rustc_macro::TokenStream) -> rustc_macro::TokenStream {
    input
}
