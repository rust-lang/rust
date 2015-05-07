// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:attr_plugin_test.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(attr_plugin_test)]
#![deny(unused_attributes)]

#[baz]
fn baz() { } // no error

#[foo]
pub fn main() {
     //~^^ ERROR unused
    #[bar]
    fn inner() {}
    //~^^ ERROR crate
    //~^^^ ERROR unused
    baz();
    inner();
}
