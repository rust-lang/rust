// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;

struct Vec2 {
    x: float,
    y: float
}

// methods we want to export as methods as well as operators
impl Vec2 {
#[inline(always)]
    fn vmul(self, other: float) -> Vec2 {
        Vec2 { x: self.x * other, y: self.y * other }
    }
}

// Right-hand-side operator visitor pattern
trait RhsOfVec2Mul<Result> { fn mul_vec2_by(&self, lhs: &Vec2) -> Result; }

// Vec2's implementation of Mul "from the other side" using the above trait
impl<Res, Rhs: RhsOfVec2Mul<Res>> Mul<Rhs,Res> for Vec2 {
    fn mul(&self, rhs: &Rhs) -> Res { rhs.mul_vec2_by(self) }
}

// Implementation of 'float as right-hand-side of Vec2::Mul'
impl RhsOfVec2Mul<Vec2> for float {
    fn mul_vec2_by(&self, lhs: &Vec2) -> Vec2 { lhs.vmul(*self) }
}

// Usage with failing inference
pub fn main() {
    let a = Vec2 { x: 3f, y: 4f };

    // the following compiles and works properly
    let v1: Vec2 = a * 3f;
    io::println(fmt!("%f %f", v1.x, v1.y));

    // the following compiles but v2 will not be Vec2 yet and
    // using it later will cause an error that the type of v2
    // must be known
    let v2 = a * 3f;
    io::println(fmt!("%f %f", v2.x, v2.y)); // error regarding v2's type
}
