// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Point {
    x: f64,
    y: f64,
}

trait NewTrait {
    fn a(&self) -> String;
}

impl NewTrait for Point {
    fn new(x: f64, y: f64) -> Point {
    //~^ ERROR method `new` is not a member of trait `NewTrait`
        Point { x: x, y: y }
    }

    fn a(&self) -> String {
        format!("({}, {})", self.x, self.y)
    }
}

fn main() {
    let p = Point::new(0.0, 0.0);
    //~^ ERROR unresolved name `Point::new`
    //~^^ ERROR failed to resolve. Use of undeclared module `Point`
    println!("{}", p.a());
}
