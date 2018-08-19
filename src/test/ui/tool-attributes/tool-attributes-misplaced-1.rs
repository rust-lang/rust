// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(tool_attributes, custom_attribute)]

type A = rustfmt; //~ ERROR expected type, found tool module `rustfmt`
type B = rustfmt::skip; //~ ERROR expected type, found tool attribute `rustfmt::skip`

#[derive(rustfmt)] //~ ERROR cannot find derive macro `rustfmt` in this scope
struct S;

#[rustfmt] // OK, interpreted as a custom attribute
fn check() {}

#[rustfmt::skip] // OK
fn main() {
    rustfmt; //~ ERROR expected value, found tool module `rustfmt`
    rustfmt!(); //~ ERROR cannot find macro `rustfmt!` in this scope

    rustfmt::skip; //~ ERROR expected value, found tool attribute `rustfmt::skip`
}
