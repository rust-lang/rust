// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the unboxed closure sugar can be used with an arbitrary
// struct type and that it is equivalent to the same syntax using
// angle brackets. This test covers only simple types and in
// particular doesn't test bound regions.

#![allow(dead_code)]
#![deny(unreachable_code)]

fn diverge() -> ! { panic!() }

fn get_u8() -> u8 {
    1
}
fn call(_: u8, _: u8) {

}
fn diverge_first() {
    call(diverge(),
         get_u8()); //~ ERROR unreachable expression
}
fn diverge_second() {
    call( //~ ERROR unreachable call
        get_u8(),
        diverge());
}

fn main() {}
