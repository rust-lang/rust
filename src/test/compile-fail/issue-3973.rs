// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
use std::io;

struct Point {
    x: float,
    y: float,
}

impl ToStr for Point { //~ ERROR implements a method not defined in the trait
    fn new(x: float, y: float) -> Point {
        Point { x: x, y: y }
    }

    fn to_str(&self) -> ~str {
        fmt!("(%f, %f)", self.x, self.y)
    }
}

fn main() {
    let p = Point::new(0.0f, 0.0f);
    io::println(p.to_str());
}
