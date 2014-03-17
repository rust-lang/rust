// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests to make sure that parens are needed for method calls without arguments.
// outputs text to make sure either an anonymous function is provided or
// open-close '()' parens are given


struct Point {
    x: int,
    y: int
}
impl Point {
    fn new() -> Point {
        Point{x:0, y:0}
    }
    fn get_x(&self) -> int {
        self.x
    }
}

fn main() {
    let point: Point = Point::new();
    let px: int =  point.get_x;//~ ERROR attempted to take value of method `get_x` on type `Point`
    //~^ NOTE maybe a missing `()` to call it? If not, try an anonymous function.
}

