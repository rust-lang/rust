// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern crate debug;

use std::gc::{Gc, GC};

struct clam {
    x: Gc<int>,
    y: Gc<int>,
}

struct fish {
    a: Gc<int>,
}

fn main() {
    let a: clam = clam{x: box(GC) 1, y: box(GC) 2};
    let b: clam = clam{x: box(GC) 10, y: box(GC) 20};
    let z: int = a.x + b.y; //~ ERROR binary operation `+` cannot be applied to type `Gc<int>`
    println!("{:?}", z);
    assert_eq!(z, 21);
    let forty: fish = fish{a: box(GC) 40};
    let two: fish = fish{a: box(GC) 2};
    let answer: int = forty.a + two.a;
    //~^ ERROR binary operation `+` cannot be applied to type `Gc<int>`
    println!("{:?}", answer);
    assert_eq!(answer, 42);
}
