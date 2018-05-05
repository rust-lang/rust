// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(exhaustive_patterns)]
#![feature(never_type)]
#![deny(unreachable_code)]

pub enum Void {}

pub fn uninhabited_parameter_i(_v: Void) {
    // A function with an uninhabited parameter
    // is permitted if its body is empty.
}

pub fn uninhabited_parameter_ii(v: !) -> i32 {
    // A function with an uninhabited parameter
    // is permitted if it simply returns a value
    // as a trailing expression to satisfy the
    // return type.
    v
}

pub fn uninhabited_parameter_iii(_v: Void, x: i32) -> i32 {
    println!("Call me if you can!"); //~^ ERROR unreachable expression
    x
}

fn main() {}
