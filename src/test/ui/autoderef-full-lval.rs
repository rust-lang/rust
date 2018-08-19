// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

struct clam {
    x: Box<isize>,
    y: Box<isize>,
}

struct fish {
    a: Box<isize>,
}

fn main() {
    let a: clam = clam{x: box 1, y: box 2};
    let b: clam = clam{x: box 10, y: box 20};
    let z: isize = a.x + b.y;
    //~^ ERROR binary operation `+` cannot be applied to type `std::boxed::Box<isize>`
    println!("{}", z);
    assert_eq!(z, 21);
    let forty: fish = fish{a: box 40};
    let two: fish = fish{a: box 2};
    let answer: isize = forty.a + two.a;
    //~^ ERROR binary operation `+` cannot be applied to type `std::boxed::Box<isize>`
    println!("{}", answer);
    assert_eq!(answer, 42);
}
