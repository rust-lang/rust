// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test

use std::io;

struct Point {
    x: f64,
    y: f64,
}

impl ToStr for Point { //~ ERROR implements a method not defined in the trait
    fn new(x: f64, y: f64) -> Point {
        Point { x: x, y: y }
    }

    fn to_str(&self) -> StrBuf {
        format!("({}, {})", self.x, self.y)
    }
}

fn main() {
    let p = Point::new(0.0, 0.0);
    println!("{}", p.to_str());
}
