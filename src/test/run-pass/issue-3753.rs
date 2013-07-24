// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #3656
// Issue Name: pub method preceeded by attribute can't be parsed
// Abstract: Visibility parsing failed when compiler parsing

use std::float;

struct Point {
    x: float,
    y: float
}

pub enum Shape {
    Circle(Point, float),
    Rectangle(Point, Point)
}

impl Shape {
    pub fn area(&self, sh: Shape) -> float {
        match sh {
            Circle(_, size) => float::consts::pi * size * size,
            Rectangle(Point {x, y}, Point {x: x2, y: y2}) => (x2 - x) * (y2 - y)
        }
    }
}

pub fn main(){
    let s = Circle(Point { x: 1f, y: 2f }, 3f);
    printfln!("%f", s.area(s));
}
