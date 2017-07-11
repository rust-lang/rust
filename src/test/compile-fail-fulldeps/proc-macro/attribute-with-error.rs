// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:attribute-with-error.rs

#![feature(proc_macro)]

extern crate attribute_with_error;

#[attribute_with_error::foo]
fn test() {
    let a: i32 = "foo";
    //~^ ERROR: mismatched types
}

fn main() {
    test();
}
