// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::{PartialEq, PartialOrd};
use std::num::NumCast;

pub trait NumExt: PartialEq + PartialOrd + Num + NumCast {}

impl NumExt for f32 {}

fn num_eq_one<T:NumExt>(n: T) {
    println!("{}", n == NumCast::from(1i).unwrap())
}

pub fn main() {
    num_eq_one(1f32); // you need to actually use the function to trigger the ICE
}
