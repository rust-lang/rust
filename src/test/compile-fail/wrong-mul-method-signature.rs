// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test is to make sure we don't just ICE if the trait
// method for an operator is not implemented properly.
// (In this case the mul method should take &f64 and not f64)
// See: #11450

struct Vec2 {
    x: f64,
    y: f64
}

impl Mul<Vec2, f64> for Vec2 {
    fn mul(&self, s: f64) -> Vec2 {
    //~^ ERROR: method `mul` has an incompatible type: expected &-ptr but found f64
        Vec2 {
            x: self.x * s,
            y: self.y * s
        }
    }
}

pub fn main() {
    Vec2 { x: 1.0, y: 2.0 } * 2.0;
}
